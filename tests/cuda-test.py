import torch

def cuda_test():
    if not torch.cuda.is_available():
        print("CUDA is not available on this system.")
        return

    device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {device_count}")

    for i in range(device_count):
        print(f"\nDevice {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Memory Total: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  Multi-Processor Count: {torch.cuda.get_device_properties(i).multi_processor_count}")
        # print(f"  Max Threads per Block: {torch.cuda.get_device_properties(i).max_threads_per_block}")
        print(f"  Max Threads per SM: {torch.cuda.get_device_properties(i).max_threads_per_multi_processor}")
        print(f"  Warp Size: {torch.cuda.get_device_properties(i).warp_size}")
        # print(f"  Clock Rate: {torch.cuda.get_device_properties(i).clock_rate / 1e3:.0f} MHz")
        # print(f"  Memory Clock Rate: {torch.cuda.get_device_properties(i).memory_clock_rate / 1e3:.0f} MHz")
        # print(f"  Memory Bus Width: {torch.cuda.get_device_properties(i).memory_bus_width} bits")

if __name__ == "__main__":
    cuda_test()
