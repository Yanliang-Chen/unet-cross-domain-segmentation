# 域偏移下的息肉分割：固定 U-Net 的无监督域适应

源域 Kvasir-SEG 有标签，目标医院/设备（CVC-ClinicDB、ETIS、CVC-ColonDB、PolypGen、FAR）只有无标签图片。本项目研究：无监督域适应能否让**原始 U-Net**（基础通道 16、4 层、输入 256，结构固定）在新域上稳定地比“继续训练”分割得更好，以及适应后的模型能否不用目标标签选出来。

每个方法都与同起点、同步数、无目标损失的**继续训练对照**配对比较，部署的检查点按源验证 Dice 事前固定的规则选择，目标标签只用于最终评价。

## 当前结果（2026-11-02）

部署 Dice（6 条源轨迹 × 3 个 λ，FAR 只用 λ=0.5；每条轨迹按源验证选检查点）：

| 方法 | ClinicDB | ETIS | ColonDB | PolypGen（正常图误报） | FAR（正常图误报） |
|---|---:|---:|---:|---:|---:|
| DACS 式跨域粘贴（分割专用） | **0.686** | 0.403 | **0.609** | 0.599（37%） | **0.565**（45%） |
| MinEnt+MMD（分割专用） | 0.670 | 0.412 | 0.568 | 0.603（40%） | 0.542（65%） |
| FDA | 0.650 | **0.430** | 0.508 | **0.611**（37%） | 0.563（54%） |
| MinEnt | 0.664 | 0.393 | 0.534 | – | – |
| MMD | 0.658 | 0.388 | 0.541 | – | – |
| 继续训练对照 | 0.637 | 0.338 | 0.435 | 0.592（79%） | 0.492（74%） |
| SourceOnly | 0.610 | 0.366 | 0.444 | 0.573（25%） | 0.510（57%） |

- 适应在四个目标上显著优于继续训练对照，PolypGen 上收益小且区间含零。相对 SourceOnly 区间不含零：MinEnt+MMD 5/5，DACS、FDA 4/5。
- 单纯继续训练会大幅增加正常图误报（PolypGen 79%、FAR 74%）；DACS 在 FAR 上把误报降到 SourceOnly 以下，PolypGen 上 λ≥0.5 时与 SourceOnly 持平。
- 最好的方法随目标域改变；DACS 在三个目标上第一，MinEnt+MMD 在所有目标上都排前三。
- 小连通域后处理（删除小于图像 0.1% 的预测块，事前固定）几乎不损 Dice（PolypGen 反而 +0.005–0.015），正常图误报 FAR 降 5–7 个百分点、PolypGen 降 4–5 个百分点，不改变方法排序。
- 无标签选模：源验证是最好的单一规则；基于置信度的验证器和 CMR 会选中自信但错误或几乎全空的模型。

完整设计、结果与局限见[项目总览与实验设计](项目总览与实验设计.md)。

## 文档

| 文件 | 内容 |
|---|---|
| [项目总览与实验设计.md](项目总览与实验设计.md) | 研究问题、数据、方法、评价、全部结果、剩余工作、局限、参考文献（报告格式） |
| [EXPERIMENT_PROPOSAL.md](EXPERIMENT_PROPOSAL.md) | 英文实验方案：假设、已完成、待做与判定规则 |
| [benchmark_protocol.yaml](benchmark_protocol.yaml) | 事前固定的场景、方法、部署规则、指标与验证器设置 |
| [results/benchmark/](results/benchmark/) | 各目标的逐候选分数（`candidates.csv`）和配对收益表（`paired.md`） |
| [每日进度.md](每日进度.md) | 每天的完成情况和下一步 |
| [失败方法记录.md](失败方法记录.md) | 2026-10-31 之前 18 类未达目标的尝试（报告附录） |

## 代码

| 入口 | 用途 |
|---|---|
| `train.py`、`select_model.py` | 源域 U-Net 训练（A0/A1 × 种子 42/43/44）与源验证选模 |
| `python -m benchmark.data` | 构建各场景的 256 图像缓存与原尺寸掩码 |
| `python -m benchmark.uda grid --scenario <场景> --algorithms <方法...>` | AdaBN + 适应方法 × λ × 6 条源轨迹；`--lams`、`--every` 控制 λ 与检查点间隔 |
| `python -m benchmark.uda train --algorithm continue --scenario <场景> --lams 0` | 继续训练对照 |
| `python -m benchmark.score run --pool <池> --scenarios source_val <场景>` | 原尺寸指标与验证器打分 |
| `python -m benchmark.analyze --pools <池...> --name <名>` | Oracle、选模差距与区间、标注预算 |
| `python -m benchmark.report --scenario <场景>` | 写入 `results/benchmark/<场景>/` 的配对收益表 |
| `python -m benchmark.postprocess` | 部署模型的小连通域后处理（`results/benchmark/postprocessing.md`） |
| `python -m benchmark.curves` | λ、训练步数曲线与小息肉 Dice（只读 `candidates.csv`，写 `results/benchmark/curves.md` 与 `figures/`） |
| `python -m benchmark.summary_figures` | 汇总图：各目标收益与区间、PolypGen/FAR 的 Dice–误报散点（`results/benchmark/figures/`） |
| `python -m benchmark.qualitative` | 按固定规则选图的定性例子（`results/benchmark/figures/qualitative_<场景>.png`） |
| `bash benchmark/run_scenario.sh <场景> <方法...>` | 一个场景的对照、适应、打分与分析全流程（环境变量 `LAMS`、`EVERY` 可选） |
| `python -m benchmark.extra --pool <池>` | 可选：瓶颈特征、CMR-TTD、MixVal 等第二轮验证器 |

方法实现集中在 [benchmark/uda.py](benchmark/uda.py)（AdaBN、MinEnt、AdvEnt、DANN、MMD、BNM、MCC、Mean Teacher、FDA、MinEnt+MMD、DACS、继续训练对照），验证器在 [benchmark/validators.py](benchmark/validators.py)。`scripts/` 和根目录下的其余协议文件属于早期工作，保留以便追溯。

## 环境

训练和评价在 Windows + CUDA 上运行（RTX 5070）。仓库不包含医学数据、权重或缓存；检查点和逐图分数在被 Git 忽略的 `runs/`，写报告需要的汇总表由 `benchmark.report` 复制到 `results/benchmark/`。

新 Windows 环境：

```powershell
.\scripts\setup_windows.ps1
```

示例：在 ETIS 上跑三个方法的完整流程并生成报告：

```bash
bash benchmark/run_scenario.sh etis dacs minent_mmd fda
.venv/Scripts/python.exe -m benchmark.report --scenario etis
```
