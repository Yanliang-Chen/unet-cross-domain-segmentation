import platform
import sys
import torch

print("Python:",sys.version.split()[0])
print("OS:",platform.platform())
print("PyTorch:",torch.__version__)
print("CUDA available:",torch.cuda.is_available())
if torch.cuda.is_available():
    print("NVIDIA GPU:",torch.cuda.get_device_name(0))
    print("PyTorch CUDA runtime:",torch.version.cuda)
print("MPS built:",torch.backends.mps.is_built())
print("MPS available:",torch.backends.mps.is_available())
