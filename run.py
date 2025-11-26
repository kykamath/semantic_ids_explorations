import torch


def hello_torch():
    # Create a simple tensor and perform a small operation
    x = torch.tensor([1, 2, 3])
    y = x * 2

    print("Hello, World — from PyTorch!")
    print("tensor x:", x)
    print("tensor y (x * 2):", y)
    print("PyTorch version:", torch.__version__)

    # Show whether CUDA is available
    try:
        if torch.cuda.is_available():
            print("CUDA available — GPU:", torch.cuda.get_device_name(0))
        else:
            print("CUDA not available; using CPU.")
    except Exception:
        # Getting device name can fail in some restricted environments
        print("CUDA availability check failed or not supported in this environment.")


if __name__ == '__main__':
    hello_torch()