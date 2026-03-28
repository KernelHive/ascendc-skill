import torch
import torch_npu
import time
import sys
import argparse

def check_npu(device='npu:0'):
    print("=== NPU Diagnostics ===")
    try:
        if not torch.npu.is_available():
            print("[ERROR] NPU is not available (torch.npu.is_available() == False)")
            return False
        
        device_count = torch.npu.device_count()
        print(f"NPU Device Count: {device_count}")
        
        # Try a simple computation
        print("Attempting simple tensor operation on NPU...")
        x = torch.ones(1024, 1024, device=device)
        y = torch.ones(1024, 1024, device=device)
        
        # Warmup
        z = x + y
        print(f"[DEBUG] Synchronize after addition")
        torch.npu.synchronize(device=device)
        print(f"[DEBUG] Synchronize after multiplication")
        start_time = time.time()
        z = x * y
        res = z.cpu() # This forces synchronization
        print(f"[DEBUG] Synchronize after result")
        torch.npu.synchronize(device=device)
        print(f"[DEBUG] Synchronize after synchronize")
        end_time = time.time()
        
        if torch.all(res.eq(1.0)):
            print(f"[SUCCESS] NPU is alive and computing correctly. Time: {end_time - start_time:.4f}s")
            return True
        else:
            print("[ERROR] NPU computation result is incorrect!")
            return False
            
    except Exception as e:
        print(f"[FATAL] NPU Check Failed with exception: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check NPU health')
    parser.add_argument('--device', type=str, default='npu:0', help='NPU device to check')
    args = parser.parse_args()
    success = check_npu(args.device)
    if not success:
        sys.exit(1)

