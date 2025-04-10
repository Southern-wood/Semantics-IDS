import argparse

parser = argparse.ArgumentParser(description='TranAD')
parser.add_argument('--dataset', type=str, default='SWaT', choices=['SWaT', 'WADI'])
parser.add_argument('--noise_type', type=str, default='pure', choices=['pure', 'noise', 'missing', 'duplicate', 'delay', 'mismatch'])
parser.add_argument('--noise_level', type=str, default='low', choices=['low', 'high'])
parser.add_argument('--num_epoch', type=int, default=1, help='number of epochs')
parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--retrain', action='store_true', help='retrain the model')
parser.add_argument('--model_save_path', type=str, default='trained_models', help='path to save the model')
args = parser.parse_args()