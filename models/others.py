# others.py
import torch

@staticmethod
def check_gpu_memory():
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert bytes to GB
            reserved_memory = torch.cuda.memory_reserved(i) / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
            free_memory = reserved_memory - allocated_memory
            print(f"  Total memory: {total_memory:.2f} GB")
            print(f"  Reserved memory: {reserved_memory:.2f} GB")
            print(f"  Allocated memory: {allocated_memory:.2f} GB")
            print(f"  Free memory: {free_memory:.2f} GB")
    else:
        print("No GPU is available.")