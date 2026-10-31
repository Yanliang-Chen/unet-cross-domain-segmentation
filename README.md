# 跨域息肉图像分割：无标签选模与完整流程评估

项目研究目标域无分割标签时，适应、选模和预测拒绝的完整流程能否优于强源域基线。当前**尚未证明完整 UDA 方法在 A0/A1 下同时改善分割与正常图误报**。

## 项目文档

- [项目总览与实验设计](项目总览与实验设计.md)：题目、研究目标、数据角色、实验设计、结果、实现状态、剩余任务和参考文献。
- [失败方法记录](失败方法记录.md)：18 类失败或退役尝试、主要失败结果及原因分析。
- [每日进度](每日进度.md)：本轮完成情况和下次继续的位置。
- [关键证据摘要](evidence/key_results.json)：冻结结果和原输出 SHA-256；查看结果无需重新训练。

## 保留的结果

| 项目 | 当前证据 |
| --- | --- |
| 强源域继续训练与等权集成 | FAR 事前列出的次要比较：A0/A1 Dice +0.0152/+0.0107；UDA 集成的主比较仍失败 |
| 冻结 SAM 外部教师学生 | ClinicDB 相对强单模型六种子均改善；A1 相对同损失 self 的独立收益未证实 |
| 局部相对存在性头 | 相对同一学生，两增强均通过开发期误报下降/分割保留门槛；尚非完整 UDA 成功 |

全部种子保留。已反复使用的 ClinicDB、PolypGen 只能用于开发；FAR 已完成一次正式评价，不再作为后续调参资料。下一步重点是直接面向整图误报的目标和新的独立确认，详见总览。

## 清理后的代码

| 入口 | 用途 |
| --- | --- |
| `train.py`、`select_model.py`、`evaluate.py` | 原 A0/A1 U-Net、源验证最佳检查点与原 256 网格评价 |
| `scripts/train_source_control.py` | 固定 BN、EMA 的强源域继续训练 |
| `scripts/build_external_teacher.py`、`train_external_teacher.py` | 冻结教师缓存与匹配 external/self 训练 |
| `scripts/evaluate_external_teacher.py`、`analyze_external_teacher.py` | 教师实验原尺寸指标与配对来源组分析 |
| `scripts/build_presence_features.py` | 冻结来源角色、构建源图/评价图的 1,024 维中间特征缓存 |
| `scripts/presence.py`、`analyze_presence.py` | 512 维局部相对特征头、评价与配对分析 |
| `models/`、`datasets/`、`scripts/artifacts.py`、`checkpoints.py` | 保留网络、配对和共享辅助函数 |
| `scripts/prepare_polyp_data.py`、`audit_real_data.py`、`check_protocol.py` | 数据准备与必要的划分保护 |

失败方法的活动入口和过期说明已删除，结论集中保留在失败记录。原始权重、逐图结果、冻结清单以及清理前源码保留在本机 `runs/`，不随 Git 分发。冻结历史协议保留原字节，`.gitattributes` 防止 Git 换行转换破坏其 SHA-256；其中 `far_confirmatory_protocol.yaml` 和 `validated_baseline_protocol.yaml` 是证据身份记录，不是当前方法菜单。

## Windows 环境与使用

训练和评价使用 Windows CUDA，Mac 用于编辑代码与 Git。仓库不包含医学数据、预训练权重或实验缓存。现有脚本依赖本机的冻结来源清单，尚未实现从空目录一键恢复全部历史研究。

已有虚拟环境无需重装；新 Windows 环境可运行：

```powershell
.\scripts\setup_windows.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements-external-teacher.txt
```

基础六组矩阵已经完成。以下仅显示入口，不会启动训练：

```powershell
.\.venv\Scripts\python.exe -m scripts.run_matrix --dry-run --device cuda
```

源验证选择器按单个 seed/augmentation 运行，输出必须是新文件：

```powershell
.\.venv\Scripts\python.exe select_model.py --runs runs/real/a0_seed42 --output runs/real/source_selection_clean/a0_seed42.json
```

强源域继续训练读取现有冻结源检查点清单，默认写入新目录 `runs/real/source_control_clean`。教师脚本保留原冻结路径，已有结果会阻止覆盖；不要为查看进度重复运行。

局部头拆成缓存构建和相对头拟合。默认复用本机已有 `runs/real/presence_gate` 的冻结特征，输出到 `runs/real/presence_clean`。以下会实际拟合并重新生成评价文件，**本次清理未运行，也无需为查看既有结论运行**：

```powershell
.\.venv\Scripts\python.exe -m scripts.presence fit
.\.venv\Scripts\python.exe -m scripts.presence evaluate
.\.venv\Scripts\python.exe -m scripts.analyze_presence
```

需要重新构建特征时，顺序为 `build_presence_features freeze` → `cache-source` → `presence fit --cache-root runs/real/presence_features_clean` → `cache-evaluation` → `presence evaluate --cache-root runs/real/presence_features_clean` → `analyze_presence`，均通过 `python -m scripts.<模块>` 调用。缓存依赖的 `negative_context/split.json`、`balanced_context/split.json` 仅承载仍有效的数据角色，不要求恢复失败训练代码。新方案应使用新目录和新协议，不改写历史冻结结果。

提交按一天主要成果汇总；本次按用户收尾授权统一提交。数据、权重、缓存、临时归档不提交。
