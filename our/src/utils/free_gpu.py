import torch
import subprocess

# This script is used to find the GPU with the most available memory using nvidia-smi.

def get_gpu_memory():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi error: {result.stderr}")
    return [tuple(map(int, x.split(', '))) for x in result.stdout.strip().split('\n')]

def get_lowest_memory_gpu():
    if not torch.cuda.is_available():
        return None
    gpu_memory = get_gpu_memory()
    free_memory = [(total - used, i) for i, (total, used) in enumerate(gpu_memory)]
    _, best_gpu = max(free_memory)
    if _ < 6000:
        raise RuntimeError("No free GPU  available")
    print(f'Best GPU: {best_gpu}')
    return best_gpu

opti_device = torch.device(f'cuda:{get_lowest_memory_gpu()}')