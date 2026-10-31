"""Small artifact helpers shared by the retained pipelines."""
import json
from pathlib import Path
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")


@torch.no_grad()
def update_teacher(teacher, student, decay):
    for target, source in zip(teacher.parameters(), student.parameters()):
        target.lerp_(source, 1-decay)
    for target, source in zip(teacher.buffers(), student.buffers()):
        target.copy_(source)
