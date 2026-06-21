import torch
import torch.nn as nn
import time
from export_model import BasicMLP

def main():
    model = BasicMLP()
    model.eval()
    
    dummy_input = torch.randn(1, 10)
    
    # Warmup
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)

    iterations = 100000
    
    print("Running PyTorch Benchmark...")
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time_ms = (total_time / iterations) * 1000
    
    print(f"--- PyTorch Benchmark ---")
    print(f"Iterations: {iterations}")
    print(f"Total Time: {total_time:.4f} seconds")
    print(f"Average Time per inference: {avg_time_ms:.6f} ms")

if __name__ == "__main__":
    main()
