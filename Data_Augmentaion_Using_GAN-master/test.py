import torch


def verify_gpu_setup():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        # Test CUDA with a simple operation
        print("\nTesting CUDA with tensor operations...")
        x = torch.rand(5, 3)
        print("CPU Tensor:")
        print(x)

        x = x.cuda()
        print("\nGPU Tensor:")
        print(x)

        print("\nGPU setup is working correctly!")
    else:
        print("CUDA is not available. Please check your installation.")


if __name__ == "__main__":
    verify_gpu_setup()