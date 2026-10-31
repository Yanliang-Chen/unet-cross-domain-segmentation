"""UDA candidate generators for the fixed U-Net; each adds checkpoints to a registry pool.

  python -m benchmark.uda adabn --scenario clinicdb
  python -m benchmark.uda train --algorithm minent --scenario clinicdb --lams 0.1 0.5 1
  python -m benchmark.uda grid --scenario clinicdb

Target images are the unlabelled validation images of the scenario (never masks).
Training methods warm-start from the run's final source checkpoint (epoch 30) and keep
the source BCE+Dice loss. Extra modules (discriminators, EMA teacher) are training-only;
the deployed network is always the unchanged U-Net.
"""
import argparse
import copy
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from benchmark.data import build_cache
from benchmark.score import REGISTRY, load_models
from benchmark.validators import MCC_TEMPERATURE, binary_entropy
from datasets.segmentation import SegmentationDataset
from scripts.artifacts import update_teacher
from train import augment
from utils import atomic_torch_save, bce_dice_loss, choose_device, seed_everything

OUT = Path("runs/benchmark/uda")
RUNS = [f"{a}_seed{s}" for a in ("a0", "a1") for s in (42, 43, 44)]
ALGORITHMS = ("minent", "advent", "dann", "mmd", "bnm", "mcc", "meanteacher", "fda")
# Matched control: same warm start, steps, batches and joint source+target forward pass
# (target images still reach BatchNorm), but no target loss term. Run with --lams 0.
CONTROLS = ("continue",)
# loss = source + lam * WEIGHT * term; FDA uses lam to set its band size beta instead
WEIGHT = {"continue": 0.0, "minent": 1.0, "advent": 0.01, "dann": 0.1, "mmd": 1.0, "bnm": 1.0, "mcc": 1.0,
          "meanteacher": 1.0}
FDA_BETA = {0.1: 0.01, 0.5: 0.05, 1.0: 0.09}
STEPS, EVERY, LR, BATCH = 2000, 200, 1e-4, 8


def target_images(scenario):
    cache = build_cache(scenario)
    keep = torch.tensor([r["role"] in ("val", "both") for r in cache["rows"]])
    return cache["images"][keep]


def register(pool, rows):
    path = REGISTRY/f"{pool}.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    merged = {r["id"]: r for r in existing}
    merged.update({r["id"]: r for r in rows})
    path.write_text(json.dumps(list(merged.values()), indent=1)+"\n", encoding="utf-8")


def candidate(algorithm, scenario, run, lam, step, checkpoint):
    augmentation, seed = run.split("_seed")
    return {"id": f"{algorithm}_{scenario}_{run}_l{lam:g}_s{step:04d}", "algorithm": algorithm,
            "lam": lam, "augmentation": augmentation, "seed": int(seed), "step": step,
            "run": f"{algorithm}_{run}_l{lam:g}", "checkpoints": [str(checkpoint)],
            "scenarios": [scenario, "source_val"]}


@torch.no_grad()
def adabn(scenario, runs, device, batch=32):
    """Re-estimate BatchNorm statistics on target images for every source checkpoint."""
    images, rows = target_images(scenario), []
    for run in runs:
        for path in sorted(Path("runs/real", run, "checkpoints").glob("epoch_*.pt")):
            epoch = int(path.stem.split("_")[1])
            out = OUT/"adabn"/scenario/run/f"epoch_{epoch:03d}.pt"
            if not out.exists():
                state = torch.load(path, map_location="cpu", weights_only=True)
                model = load_models([path], device)[0]
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats()
                        m.momentum = None                 # cumulative average over all batches
                model.train()
                for start in range(0, len(images), batch):
                    model(images[start:start+batch].to(device).float()/255)
                atomic_torch_save({"config": state["config"], "model_state_dict": model.state_dict(),
                                   "source_checkpoint": str(path)}, out)
            rows.append(candidate("adabn", scenario, run, 0.0, epoch, out))
    register("uda_adabn", rows)
    print("adabn", scenario, len(rows), flush=True)


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale*grad, None


def output_discriminator():
    """AdvEnt-style fully convolutional discriminator on 2-channel self-information maps."""
    layers, channels = [], [2, 32, 64, 128, 256]
    for a, b in zip(channels, channels[1:]):
        layers += [nn.Conv2d(a, b, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
    return nn.Sequential(*layers, nn.Conv2d(256, 1, 4, 2, 1))


def self_information(logits):
    p = logits.sigmoid().clamp(1e-6, 1-1e-6)
    probs = torch.cat((1-p, p), 1)
    return -probs*probs.log()/torch.log(torch.tensor(2.0))


def gaussian_mmd(a, b):
    x = torch.cat((a, b))
    distance = torch.cdist(x, x).pow(2)
    bandwidth = distance.detach()[distance > 0].median()
    kernel = sum(torch.exp(-distance/(bandwidth*s)) for s in (0.5, 1.0, 2.0))
    n = len(a)
    return kernel[:n, :n].mean() + kernel[n:, n:].mean() - 2*kernel[:n, n:].mean()


def pixel_matrix(logits, pixels, generator, temperature=1.0):
    p = (logits/temperature).sigmoid().flatten()
    index = torch.randint(len(p), (pixels,), generator=generator, device=p.device)
    p = p[index]
    return torch.stack((1-p, p), 1)


def mcc_loss(logits, generator, pixels=4096):
    y = pixel_matrix(logits, pixels, generator, MCC_TEMPERATURE)
    h = -(y*y.clamp_min(1e-6).log()).sum(1)
    w = 1 + torch.exp(-h)
    w = len(w)*w/w.sum()
    confusion = y.t() @ (w[:, None]*y)
    confusion = confusion/confusion.sum(1, keepdim=True)
    return (confusion.sum() - confusion.trace())/2


def fda_source_to_target(source, target, beta):
    """Replace the low-frequency amplitude of source images with that of target images."""
    fs, ft = torch.fft.fft2(source), torch.fft.fft2(target)
    amplitude, phase = fs.abs(), fs.angle()
    amplitude_t = ft.abs()
    h, w = source.shape[-2:]
    b = max(1, int(min(h, w)*beta))
    shifted = torch.fft.fftshift(amplitude, dim=(-2, -1))
    shifted_t = torch.fft.fftshift(amplitude_t, dim=(-2, -1))
    ch, cw = h//2, w//2
    shifted[..., ch-b:ch+b, cw-b:cw+b] = shifted_t[..., ch-b:ch+b, cw-b:cw+b]
    amplitude = torch.fft.ifftshift(shifted, dim=(-2, -1))
    return torch.fft.ifft2(amplitude*torch.exp(1j*phase)).real.clamp(0, 1)


def train(algorithm, scenario, run, lam, device, steps=STEPS, every=EVERY):
    folder = OUT/algorithm/scenario/f"{run}_l{lam:g}"
    rows = [candidate(algorithm, scenario, run, lam, s, folder/f"step_{s:04d}.pt")
            for s in range(every, steps+1, every)]
    if all(Path(r["checkpoints"][0]).exists() for r in rows):
        return rows
    images = target_images(scenario)
    source = SegmentationDataset("data/polyp_png", "source_train", 256, 3)
    start = Path("runs/real", run, "checkpoints", "epoch_030.pt")
    cfg = torch.load(start, map_location="cpu", weights_only=True)["config"]
    seed = int(cfg["seed"])
    seed_everything(seed)
    model = load_models([start], device)[0].train()
    features = {}
    model.bottleneck.register_forward_hook(lambda m, i, o: features.__setitem__("z", o))
    extra = nn.ModuleDict()
    if algorithm == "advent":
        extra["d"] = output_discriminator()
    if algorithm == "dann":
        extra["d"] = nn.Sequential(nn.Linear(model.bottleneck[4].num_features, 128),
                                   nn.ReLU(inplace=True), nn.Linear(128, 1))
    extra = extra.to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + (list(extra.parameters()) if algorithm == "dann" else []),
                                  lr=LR, weight_decay=float(cfg["weight_decay"]))
    d_optimizer = torch.optim.Adam(extra.parameters(), lr=LR, betas=(0.9, 0.99)) if algorithm == "advent" else None
    teacher = copy.deepcopy(model).eval() if algorithm == "meanteacher" else None
    loader = DataLoader(source, batch_size=BATCH, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed+1000))
    geometry = torch.Generator().manual_seed(seed+2000)
    photometric = torch.Generator().manual_seed(seed+3000)
    target_rng = torch.Generator().manual_seed(seed+4000)
    pixel_rng = torch.Generator(device=device).manual_seed(seed+5000)
    step, iterator = 0, iter(loader)
    while step < steps:
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x, y = augment(x, y, cfg["augmentation"], geometry, photometric)
        x, y = x.to(device), y.to(device)
        t = images[torch.randint(len(images), (BATCH,), generator=target_rng)].to(device).float()/255
        if algorithm == "fda":
            x = fda_source_to_target(x, t, FDA_BETA[lam])
            loss = bce_dice_loss(model(x), y)
        else:
            logits = model(torch.cat((x, t)))
            zs, zt = features["z"].mean((2, 3)).split(BATCH)
            ls, lt = logits.split(BATCH)
            loss, weight = bce_dice_loss(ls, y), lam*WEIGHT[algorithm]
            if algorithm == "minent":
                loss = loss + weight*binary_entropy(lt.sigmoid()).mean()
            elif algorithm == "advent":
                for p in extra.parameters():
                    p.requires_grad_(False)
                out = extra["d"](self_information(lt))
                loss = loss + weight*F.binary_cross_entropy_with_logits(out, torch.ones_like(out))
            elif algorithm == "dann":
                domain = extra["d"](GradientReverse.apply(torch.cat((zs, zt)), weight)).squeeze(1)
                labels = torch.cat((torch.ones(BATCH), torch.zeros(BATCH))).to(device)
                loss = loss + F.binary_cross_entropy_with_logits(domain, labels)
            elif algorithm == "mmd":
                loss = loss + weight*gaussian_mmd(zs, zt)
            elif algorithm == "bnm":
                a = pixel_matrix(lt, 4096, pixel_rng)
                loss = loss - weight*torch.linalg.svdvals(a).sum()/len(a)
            elif algorithm == "mcc":
                loss = loss + weight*mcc_loss(lt, pixel_rng)
            elif algorithm == "meanteacher":
                noise = torch.randn(t.shape, generator=pixel_rng, device=device)*0.05
                student = model((t+noise).clamp(0, 1)).sigmoid()
                with torch.no_grad():
                    target_p = teacher(t).sigmoid()
                loss = loss + weight*F.mse_loss(student, target_p)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if algorithm == "advent":
            for p in extra.parameters():
                p.requires_grad_(True)
            ds = extra["d"](self_information(ls.detach()))
            dt = extra["d"](self_information(lt.detach()))
            d_loss = (F.binary_cross_entropy_with_logits(ds, torch.ones_like(ds))
                      + F.binary_cross_entropy_with_logits(dt, torch.zeros_like(dt)))/2
            d_optimizer.zero_grad(set_to_none=True)
            d_loss.backward()
            d_optimizer.step()
        if teacher is not None:
            update_teacher(teacher, model, 0.99)
        step += 1
        if step % every == 0:
            deployed = teacher if teacher is not None else model
            atomic_torch_save({"config": cfg, "model_state_dict": deployed.state_dict(),
                               "source_checkpoint": str(start), "algorithm": algorithm,
                               "lam": lam, "step": step}, folder/f"step_{step:04d}.pt")
    print(algorithm, scenario, run, f"lam={lam:g}", "final loss", round(float(loss.detach()), 4), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["adabn", "train", "grid"])
    parser.add_argument("--algorithm", choices=ALGORITHMS + CONTROLS)
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS))
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--runs", nargs="+", default=RUNS)
    parser.add_argument("--lams", nargs="+", type=float, default=[0.1, 0.5, 1.0])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = choose_device(args.device)
    if args.command == "adabn":
        adabn(args.scenario, args.runs, device)
        return
    algorithms = [args.algorithm] if args.command == "train" else args.algorithms
    if args.command == "grid":
        adabn(args.scenario, args.runs, device)
    for run in args.runs:                     # run-major: every algorithm appears early
        for lam in args.lams:
            for algorithm in algorithms:
                register(f"uda_{algorithm}", train(algorithm, args.scenario, run, lam, device))


if __name__ == "__main__":
    main()
