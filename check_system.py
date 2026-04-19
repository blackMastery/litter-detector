# check_system.py
import torch
import psutil
import shutil
import platform

print("\n── System Check ─────────────────────────────")
print(f"  OS          : {platform.system()} {platform.release()}")
print(f"  Python      : {platform.python_version()}")
print(f"  CPU cores   : {psutil.cpu_count(logical=False)} physical  /  {psutil.cpu_count()} logical")

ram = psutil.virtual_memory()
print(f"  RAM         : {ram.total / 1e9:.1f} GB total  |  {ram.available / 1e9:.1f} GB available")

disk = shutil.disk_usage(".")
print(f"  Disk free   : {disk.free / 1e9:.1f} GB")

print(f"\n── PyTorch / GPU ────────────────────────────")
print(f"  PyTorch     : {torch.__version__}")
print(f"  CUDA avail  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram  = props.total_memory / 1e9
        print(f"  GPU {i}        : {props.name}")
        print(f"  VRAM        : {vram:.1f} GB")
        print(f"  CUDA ver    : {torch.version.cuda}")
    print(f"\n  ✔ GPU training available")
else:
    print(f"\n  ○ No GPU found — will train on CPU (slower)")

print()