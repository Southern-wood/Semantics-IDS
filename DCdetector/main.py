import os

import torch.cuda
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
from generate_testfiles import *
import argparse
import numpy as np
from torch.backends import cudnn
from utils.utils import *
from solver import Solver
import time
import warnings
import datetime
import random
warnings.filterwarnings('ignore')

import torch
import subprocess

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
        raise RuntimeError("No free GPU available")
    print(f'Best GPU: {best_gpu}')
    return best_gpu

import sys

class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass


def str2bool(v):
    return v.lower() in ('true')


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(array[idx-1])


def main(config):
    # if not torch.cuda.is_available():
    #     raise RuntimeError("CUDA is not available. Please check your CUDA installation.")
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        os.mkdir(config.model_save_path)
    
    prefix = "../processed"

    attack_path_list = generate_testfiles('../processed', config.dataset)
    if config.attack_path is None:
        config.attack_path = attack_path_list[0]
    normal_path, labels_path = generate_trainpath_and_label(prefix, config.dataset, config.quality_type, config.level)
    # sys.stdout = Logger("result/test.log", sys.stdout)
    print("Training on: ", config.normal_path)

    config.normal_path = normal_path
    config.labels_path = labels_path
    config.attack_path = attack_path_list[0]        
    config.train_path = config.quality_type + '_' + str(config.level)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('================ Hyperparameters ===============')
    for k, v in args.items():
        print('%s: %s' % (str(k), str(v)))

    if config.mode == 'train':    
        print('====================  Train  ===================')
        solver = Solver(vars(config))
        solver.train()        
    elif config.mode == 'test':
        print('====================  Test  ===================')
        for attack_path in attack_path_list:
            config.attack_path = attack_path
            solver = Solver(vars(config))
            solver.test()
    else:
        print('====================  Train  ===================')
        solver = Solver(vars(config))
        solver.train()
        print('====================  Test  ===================')
        for attack_path in attack_path_list:
            config.attack_path = attack_path
            solver = Solver(vars(config))
            solver.test()
        
    # return solver

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Alternative
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--patch_size', type=list, default=[5])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss_fuc', type=str, default='MSE')
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--rec_timeseries', action='store_true', default=True)


    parser.add_argument('--quality_type', type=str, default='quality', choices=['pure', 'noise', 'missing', 'duplicate', 'delay', 'mismatch', 'mix'])
    parser.add_argument('--level', type=str, default='low', choices=['low', 'high'])

    # Path settings (Auto generated)
    parser.add_argument('--attack_path', type=str, default=None)
    parser.add_argument('--normal_path', type=str, default=None)
    parser.add_argument('--labels_path', type=str, default=None)

    # GPU settings
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=4, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    parser.add_argument('--devices', type=str, default='0,1,2,3,4',help='device ids of multile gpus')

    # Default
    parser.add_argument('--index', type=int, default=137)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--input_c', type=int, default=9)
    parser.add_argument('--output_c', type=int, default=9)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'mix'])
    # parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')

    parser.add_argument('--anormly_ratio', type=float, default=4.00)

    config = parser.parse_args()
    args = vars(config)

    config.patch_size = [int(patch_index) for patch_index in config.patch_size]
    
    config.use_gpu = True if torch.cuda.is_available() and config.use_gpu else False
    if config.use_gpu and config.use_multi_gpu:
        config.devices = config.devices.replace(' ','')
        device_ids = config.devices.split(',')
        config.device_ids = [int(id_) for id_ in device_ids]
        config.gpu = config.device_ids[0]

    config.gpu = torch.device(f'cuda:{get_lowest_memory_gpu()}')
    torch.cuda.set_device(config.gpu)
    torch.cuda.empty_cache()
    print('Using GPU: ', config.gpu)
    main(config)
    success = True


