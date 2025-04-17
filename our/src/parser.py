import argparse

parser = argparse.ArgumentParser(description='TranAD')
parser.add_argument('--dataset', type=str, default='SWaT', choices=['SWaT', 'WADI', 'HAI'])
parser.add_argument('--quality_type', type=str, default='pure', choices=['pure', 'noise', 'missing', 'duplicate', 'delay', 'mismatch'])
parser.add_argument('--level', type=str, default='low', choices=['low', 'high'])
parser.add_argument('--num_epoch', type=int, default=None, help='number of epochs')
parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
parser.add_argument('--batch_size', type=int, default=None, help='batch size')
parser.add_argument('--retrain', action='store_true', help='retrain the model')
parser.add_argument('--threshold', type=int, default=None, help='threshold for anomaly detection')
parser.add_argument('--model_save_path', type=str, default='trained_models', help='path to save the model')
args = parser.parse_args()