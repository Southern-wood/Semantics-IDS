import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
import functools
import os

def setup():
    if not dist.is_initialized(): 
        dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def fsdp_wrapper_model(model):
    """Wrap the model with FSDP."""
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)
    policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1000)
    fsdp_model = FSDP(model,auto_wrap_policy=policy,device_id=device,use_orig_params=True)
    return fsdp_model

def create_logger():
    import logging
    import torch.distributed as dist

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if not dist.is_initialized() or dist.get_rank() == 0:
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt='[%(levelname)s %(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.addHandler(logging.NullHandler())

    return logger
