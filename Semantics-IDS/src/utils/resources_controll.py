import torch
import subprocess
import os

# This script is used to find the GPU with the most available memory using nvidia-smi.

def set_cpu_limits(num):
    """
    Set CPU limits for the process.
    :param num: Number of threads to set for various libraries.
    """
    os.environ["OMP_NUM_THREADS"] = str(num)  # OpenMP thread
    os.environ["OPENBLAS_NUM_THREADS"] = str(num)  # OpenBLAS thread
    os.environ["MKL_NUM_THREADS"] = str(num)  # MKL thread
    os.environ["NUMEXPR_NUM_THREADS"] = str(num)  # NumExpr thread

def get_optimal_device():
    """
    Get the optimal device for PyTorch.
    :return: Device object for the optimal device(GPU with most free memory or CPU).
    """
    gpu_id = get_lowest_memory_gpu()
    if gpu_id is not None:
        opti_device = torch.device(f'cuda:{gpu_id}')
    else:
        opti_device = torch.device('cpu')
    return opti_device


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
    if _ < 12000:
        print("Warning: No GPU with sufficient memory available.")
        print("Using CPU instead.")
        return None
    # print(f'Best GPU: {best_gpu}'
    print(f'Using GPU: {best_gpu} with {free_memory[best_gpu][0]}MB free memory')
    return best_gpu



