from src.parser import args
from src.utils.free_gpu import opti_device

class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Threshold parameters for spot
lm_d = {
	'SWaT': (0.993, 1),
	'WADI': (0.999, 1),
    'HAI': (0.99995, 1.06),
}


lm = lm_d[args.dataset]

# Hyperparameters
lr_d = {
	'SWaT': 0.008, 
	'WADI': 0.0001, 
    'HAI': 0.0001,
	}

lr = lr_d[args.dataset]

batch_size_d = {
    'SWaT': 128,
    'WADI': 32,
    'HAI': 128,
}

if args.batch_size is None:
	args.batch_size = batch_size_d[args.dataset]

num_epoch_d = {
    'SWaT': 3,
    'WADI': 1,
    'HAI': 1,
}

if args.num_epoch is None:
    args.num_epoch = num_epoch_d[args.dataset]

device = opti_device
