import argparse

parser = argparse.ArgumentParser(description='TranAD')
parser.add_argument('--dataset', type=str, default='SWaT', choices=['SWaT', 'WADI', 'HAI'])
parser.add_argument('--quality_type', type=str, default='pure')
parser.add_argument('--level', type=str, default='low', choices=['low', 'high'])
parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])

# Default values for the dataset, set in the constants.py file
parser.add_argument('--data_path', type=str, default='../processed', help='path to the dataset')
parser.add_argument('--feat_num', type=int, default=None, help='number of features')
parser.add_argument('--batch_size', type=int, default=None, help='batch size')
parser.add_argument('--num_epoch', type=int, default=None, help='number of epochs')
parser.add_argument('--model_save_path', type=str, default='trained_models', help='path to save the model')
parser.add_argument('--feature_selection_batch_size', type=int, default=None, help='batch size for feature selection')
parser.add_argument('--feature_selection_num_epoch', type=int, default=None, help='number of epochs for feature selection')
parser.add_argument('--minimum_selected_features', type=float, default=None, help='minimum selected features for feature selection')
parser.add_argument('--relability_rate', type=int, default=None, help='feature selection reliability rate')
parser.add_argument('--target_test_data', type=str, default=None, help='If set, the model will be tested on this dataset')

# Resource allocation
parser.add_argument('--cpu_limit', type=int, default=32, help='number of CPU cores to use')
parser.add_argument('--gpu', default='auto', type=str, help='GPU to use (default: auto)')
args = parser.parse_args()