import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version (PyTorch built with): {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Current GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    try:
        cudnn_version = torch.backends.cudnn.version()
        print(f"CuDNN Version: {cudnn_version}")
    except Exception as e:
        print(f"Could not get CuDNN version: {e}")
else:
    print("CUDA is not available according to PyTorch.")
    # Attempt to get more build info if CUDA is not available
    try:
        if hasattr(torch.version, 'cuda') and torch.version.cuda is not None:
            print(f"PyTorch was built with CUDA version: {torch.version.cuda}")
        else:
            print("PyTorch build does not seem to include CUDA support (torch.version.cuda is None or not present).")
        if hasattr(torch.version, 'git_version'):
             print(f"PyTorch Git Version: {torch.version.git_version}")
    except Exception as e:
        print(f"Error getting detailed PyTorch version info: {e}")