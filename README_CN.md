# U-Net Windows + MacBook 双平台项目

此仓库是 **U-Net 医学图像二分类分割的可运行基线**。还没有实现跨域适应、域泛化、无标签模型选择。首次提供的圆形合成数据仅用于验证软件环境，**不能用来产生科研结论**。

## 目录
- `models/unet.py`：标准轻量 U-Net
- `datasets/segmentation.py`：8-bit 图像及二值 mask 加载
- `train.py`：训练、验证、保存 best/latest checkpoint
- `evaluate.py`：测试集 Dice、IoU
- `config.yaml`：超参数
- `scripts/check_device.py`：检测 CUDA 和 MPS
- `scripts/make_demo_data.py`：生成演示数据

## Windows NVIDIA 安装（PowerShell）
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements-common.txt
```
随后按 https://pytorch.org/get-started/locally/ 安装与 NVIDIA GPU 及驱动适配的 PyTorch CUDA 版本。

```powershell
.\.venv\Scripts\python.exe scripts/check_device.py
.\.venv\Scripts\python.exe scripts/make_demo_data.py
.\.venv\Scripts\python.exe train.py --epochs 1
.\.venv\Scripts\python.exe evaluate.py --checkpoint runs/unet_baseline/latest.pt
```

## Apple Silicon Mac 安装（Terminal）
```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -r requirements-common.txt
./.venv/bin/python -m pip install torch
./.venv/bin/python scripts/check_device.py
./.venv/bin/python scripts/make_demo_data.py
./.venv/bin/python train.py --epochs 1
./.venv/bin/python evaluate.py --checkpoint runs/unet_baseline/latest.pt
```

## 同步规则
代码同步用 GitHub。数据集和权重分别保留本地副本；只在训练停止、文件完整写入后同步到学校允许使用的云盘。GitHub 的 `.gitignore` 排除了 `data/`、`runs/`、`.venv/`、`*.pt` 等文件。科研数据应符合许可与隐私要求。

继续训练：`python train.py --resume runs/unet_baseline/latest.pt --epochs 20`，其中 20 是训练结束时的 **总 epoch 数**。CUDA 与 MPS 的结果不保证完全相同。

## 真实数据
目录为 `data/my_dataset/{train,val,test}/{images,masks}/`。图像和 mask 按文件名匹配，mask 为二值；必须按患者划分数据，避免不同图像/切片泄漏到验证或测试集。DICOM、NIfTI、16-bit TIFF 需要另外实现合适的医学影像预处理，不应直接使用当前 8-bit 数据加载器。
