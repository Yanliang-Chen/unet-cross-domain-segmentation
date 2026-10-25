param(
    [string]$PythonExe = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
)
$ErrorActionPreference = "Stop"
$ProjectPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ProjectPython -PathType Leaf)) {
    if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
        throw "Python 3.11 not found at $PythonExe. Pass -PythonExe with your Python 3.10-3.12 path."
    }
    & $PythonExe -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create .venv" }
}
& $ProjectPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed; check internet access" }
# Official Windows CUDA 12.8 build. No global Anaconda or system Python changes.
& $ProjectPython -m pip install "torch==2.11.0+cu128" --index-url "https://download.pytorch.org/whl/cu128" --extra-index-url "https://pypi.org/simple"
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed; check internet access" }
& $ProjectPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Project requirements installation failed" }
& $ProjectPython -m scripts.check_device
if ($LASTEXITCODE -ne 0) { throw "Device check failed" }
& $ProjectPython -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in the new environment'; print('CUDA ready:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "CUDA build installed, but torch cannot access the GPU" }
Write-Host "VS Code: Python: Select Interpreter -> .venv\Scripts\python.exe"
