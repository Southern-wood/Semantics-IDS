import argparse
import toml
import shutil

from preprocess import *
from data_load import *

def parse_args():
  parser = argparse.ArgumentParser(description = "Preprocess datasets")
  parser.add_argument('--dataset', '-d',required =True, choices=['SWaT', 'WADI', 'HAI'],  help='Dataset to preprocess')
  parser.add_argument('--type', '-t', required=True, 
                      choices=['pure', 'noise', 'missing', 'duplicate', 'delay', 'mismatch', 'mix_1', 'mix_2'], 
                      nargs='+', help='Types of preprocessing to apply')
  return parser.parse_args()

def load_dataset(dataset, path):
  if dataset == 'SWaT':
    return load_SWaT(path)
  elif dataset == 'WADI':
    return load_WADI(path)
  elif dataset == 'HAI':
    return load_HAI(path)
  else:
    raise ValueError("Invalid dataset name. Choose from ['SWaT', 'WADI', 'HAI'].")

def save_dataset(train, test, labels, dataset, low_quaulity_type, level, output_path):
  dataset_folder = f"{output_path}/{dataset}/"
  os.makedirs(dataset_folder, exist_ok=True)

  if low_quaulity_type == 'pure':
    train_file_name = f"{dataset}_{low_quaulity_type}.npy"
    test_file_name = f"{dataset}_{low_quaulity_type}.npy"
    labels_file_name = f"{dataset}_{low_quaulity_type}_labels.npy"
  else:
    train_file_name = f"{dataset}_{low_quaulity_type}_{level}.npy"
    test_file_name = f"{dataset}_{low_quaulity_type}_{level}.npy"
    labels_file_name = f"{dataset}_{low_quaulity_type}_{level}_labels.npy"

  os.makedirs(dataset_folder + 'train', exist_ok=True)
  os.makedirs(dataset_folder + 'test', exist_ok=True)

  np.save(os.path.join(dataset_folder, 'train', train_file_name), train)
  np.save(os.path.join(dataset_folder, 'test', test_file_name), test)
  np.save(os.path.join(dataset_folder, 'test', labels_file_name), labels)


def random_select_columns(num, column_list):
  return random.sample(list(column_list), num)

def generate_noise(dataset, train, test, labels, cate, config, low_quality_type):

  if low_quality_type == 'pure':
    save_dataset(train, test, labels, dataset, low_quality_type, None, config['output_folder'])

  # If the low quality type is single-type, load the corresponding configuration
  if low_quality_type in ['noise', 'missing', 'duplicate', 'delay', 'mismatch']:
    type_config = config['low_quality_types'][low_quality_type]
  
  for level in type_config:
    # Every low quality type has its own feature ratio (rate of features to be selected)
    if low_quality_type == 'noise':
      # gaussian noise need to be added to numerical columns
      feature_ratio = level['feature_ratio']
      intensity = level['intensity']
      numeric_columns = [col for col in train.columns if col not in cate]
      target_columns = random_select_columns(int(len(numeric_columns) * feature_ratio), numeric_columns)
      train = noise(train, target_columns, intensity)
      test = noise(test, target_columns, intensity)

    elif low_quality_type == 'missing':
      feature_ratio = level['feature_ratio']
      missing_ratio = level['missing_ratio']
      target_columns = random_select_columns(int(len(train.columns) * feature_ratio), train.columns)
      train = missing(train, target_columns, missing_ratio)
      test = missing(test, target_columns, missing_ratio)
    
    elif low_quality_type == "duplicate":
      feature_ratio = level['feature_ratio']
      duplicate_ratio = level['duplicate_ratio']
      duplicate_length = level['duplicate_length']
      target_columns = random_select_columns(int(len(train.columns) * feature_ratio), train.columns)
      train = duplicate(train, target_columns, duplicate_ratio, duplicate_length)
      test = duplicate(test, target_columns, duplicate_ratio, duplicate_length)

    elif low_quality_type == "delay":
      feature_ratio = level['feature_ratio']
      delay_length = level['delay_length']
      target_columns = random_select_columns(int(len(train.columns) * feature_ratio), train.columns)
      train = delay(train, target_columns, delay_length)
      test = delay(test, target_columns, delay_length)

    elif low_quality_type == "mismatch":
      feature_ratio = level['feature_ratio']
      relative_frequency = level['relative_frequency']
      target_columns = random_select_columns(int(len(train.columns) * feature_ratio), train.columns)
      train = mismatch(train, target_columns, relative_frequency)
      test = mismatch(test, target_columns, relative_frequency)

    elif low_quality_type == "mix_1":
      # Mix of noise and mismatch
      target_columns_list = []

      # Select a random subset of numeric columns for noise
      feature_ratio = config['low_quality_types']['noise'][level]['feature_ratio']      
      numeric_columns = [col for col in train.columns if col not in cate]
      target_columns_list.append(random_select_columns(int(len(numeric_columns) * feature_ratio), numeric_columns))
      
      # Select a random subset of all columns for mismatch
      feature_ratio = config['low_quality_types']['mismatch'][level]['feature_ratio']
      target_columns_list.append(random_select_columns(int(len(train.columns) * feature_ratio), train.columns))

      intensity = config['low_quality_types']['noise'][level]['intensity']
      relative_frequency = config['low_quality_types']['mismatch'][level]['relative_frequency']

      # Apply noise to the selected numeric columns
      train = mix_1(train, target_columns_list, intensity, relative_frequency)
      test = mix_1(test, target_columns_list, intensity, relative_frequency)

    elif low_quality_type == "mix_2":
      # Mix of missing, duplicate, and delay
      target_columns_list = []

      # Select a random subset of all columns for missing
      feature_ratio = config['low_quality_types']['missing'][level]['feature_ratio']
      target_columns_list.append(random_select_columns(int(len(train.columns) * feature_ratio), train.columns))

      # Select a random subset of all columns for duplicate
      feature_ratio = config['low_quality_types']['duplicate'][level]['feature_ratio']
      target_columns_list.append(random_select_columns(int(len(train.columns) * feature_ratio), train.columns))

      # Select a random subset of all columns for delay
      feature_ratio = config['low_quality_types']['delay'][level]['feature_ratio']
      target_columns_list.append(random_select_columns(int(len(train.columns) * feature_ratio), train.columns))

      missing_ratio = config['low_quality_types']['missing'][level]['missing_ratio']
      duplicate_ratio = config['low_quality_types']['duplicate'][level]['duplicate_ratio']
      duplicate_length = config['low_quality_types']['duplicate'][level]['duplicate_length']
      delay_length = config['low_quality_types']['delay'][level]['delay_length']

      # Apply missing, duplicate, and delay to the selected columns
      train = mix_2(train, target_columns_list, missing_ratio, duplicate_ratio, duplicate_length, delay_length)
      test = mix_2(test, target_columns_list, missing_ratio, duplicate_ratio, duplicate_length, delay_length)
      

    # Save the modified dataset
    save_dataset(train, test, labels, dataset, low_quality_type, level['level'], config['output_folder'])

    

if __name__ == '__main__':
  argparses = parse_args()
  dataset = argparses.dataset
  types = argparses.type

  print(f"Dataset: {dataset}")
  print(f"Types: {types}")

  # Load the configuration file
  with open('noise_setting.toml', 'r') as f:
    config = toml.load(f)

  root_path = config['data_folder']
  train, test, labels, categorical_column = load_dataset(dataset, root_path)
  for low_quality_type in types:
    print(f"Generating low quality data for {low_quality_type}")
    # Generate low quality data for the specified type
    generate_noise(dataset, train, test, labels, categorical_column, config, low_quality_type)