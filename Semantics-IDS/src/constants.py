from src.parser import args
import os

# If TQDM_DISABLE environment variable is set, disable colors
if os.environ.get('TQDM_DISABLE'):
    class color:
        HEADER = ''
        BLUE = ''
        GREEN = ''
        RED = ''
        FAIL = ''
        ENDC = ''
        BOLD = ''
        UNDERLINE = ''
else:
    # Keep the original color class definition
    class color:
        HEADER = '\033[95m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        RED = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

# Reorganize hyperparameters by dataset
dataset_config = {
    'SWaT': {
        'feat_num': 46,
        'lm': (0.993, 1),
        'lr': 0.01,
        'batch_size': 256,
        'num_epoch': 3,
        'feature_selection_batch_size': 64,
        'feature_selection_num_epoch': 3,
        'minimum_selected_features': 0.90,
        'relability_rate': 80,
    },
    'WADI': {
        'feat_num': 98,
        'lm': (0.999, 1),
        'lr': 0.001,
        'batch_size': 64,
        'num_epoch': 1,
        'feature_selection_batch_size': 32,
        'feature_selection_num_epoch': 3,
        'minimum_selected_features': 0.85,
        'relability_rate': 90,
    },
    'HAI': {
        'feat_num': 50,
        'lm': (0.99995, 1),
        'lr': 0.001,
        'batch_size': 128,
        'num_epoch': 3,
        'feature_selection_batch_size': 64,
        'feature_selection_num_epoch': 3,
        'minimum_selected_features': 0.90,
        'relability_rate': 75,
    }
}

args_to_set_from_default = [
    'feat_num',
    'batch_size',
    'num_epoch',
    'feature_selection_batch_size',
    'feature_selection_num_epoch',
    'minimum_selected_features',
    'relability_rate'
]

# Get configuration for the current dataset
current_config = dataset_config[args.dataset]

# Set threshold parameters for spot
lm = current_config['lm']

# Set learning rate
lr = current_config['lr']

for arg in args_to_set_from_default:
    if getattr(args, arg) is None:
        setattr(args, arg, current_config[arg])
    else:
        print(f"Using user-defined value for {arg}: {getattr(args, arg)}")