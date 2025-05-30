import argparse
import toml

from preprocess import *
from data_load import *

def parse_args():
  parser = argparse.ArgumentParser(description = "Preprocess datasets")
  parser.add_argument('--dataset', '-d',required =True, choices=['SWaT', 'WADI', 'HAI'],  help='Dataset to preprocess')
  parser.add_argument('--type', '-t', required=True,  
                      nargs='+', help='Types of preprocessing to apply')
  parser.add_argument('--vaildate', '-v', action='store_true', help='Whether to validate the dataset')
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

def save_dataset(train, test, labels, dataset, noise_type, level, output_path, validate=False, validation_ratio=0.1):
  dataset_folder = f"{output_path}/{dataset}/"
  os.makedirs(dataset_folder, exist_ok=True)

  if noise_type == 'pure':
    train_file_name = f"{dataset}_{noise_type}.npy"
    test_file_name = f"{dataset}_{noise_type}.npy"
    labels_file_name = f"{dataset}_{noise_type}_labels.npy"
  else:
    train_file_name = f"{dataset}_{noise_type}_{level}.npy"
    test_file_name = f"{dataset}_{noise_type}_{level}.npy"
    labels_file_name = f"{dataset}_{noise_type}_{level}_labels.npy"

  os.makedirs(dataset_folder + 'train', exist_ok=True)
  os.makedirs(dataset_folder + 'test', exist_ok=True)

  np.save(os.path.join(dataset_folder, 'train', train_file_name), train)
  
  if validate:
    # Split the test set into validation and test sets
    vaidation_size = int(len(test) * validation_ratio)
    val_set = test[:vaidation_size]
    test_set = test[vaidation_size:]
    labels = labels[vaidation_size:]

    # Save the validation and test sets
    np.save(os.path.join(dataset_folder, 'test', test_file_name.replace('.npy', '_val.npy')), val_set)
    np.save(os.path.join(dataset_folder, 'test', test_file_name), test_set)
    np.save(os.path.join(dataset_folder, 'test', labels_file_name), labels)
  else:
    np.save(os.path.join(dataset_folder, 'test', test_file_name), test)
    np.save(os.path.join(dataset_folder, 'test', labels_file_name), labels)


def random_select_columns(num, column_list):
  return random.sample(list(column_list), num)

def generate_noise(dataset, train, test, labels, cate, config, noise_type):

  if noise_type == 'pure':
    save_dataset(train, test, labels, dataset, noise_type, None, config['output_folder'], config['validate'], config['validation_ratio'])

  # If the noise type is single-type, load the corresponding configuration
  if noise_type in ['noise', 'missing', 'duplicate', 'delay']:
    type_config = config['noise_types'][noise_type]  

    # Note we do not change the original dataset in single noise settings, so no need to copy the original dataset

    for level in ['low', 'high']:
      parameter = type_config[level]
      # Every noise type has its own feature ratio (rate of features to be selected)
      feature_ratio = parameter['feature_ratio']
      if noise_type == 'noise':  
        # gaussian noise need to be added to numerical columns
        intensity = parameter['intensity']
        numeric_columns = [col for col in train.columns if col not in cate]
        target_columns = random_select_columns(int(len(numeric_columns) * feature_ratio), numeric_columns)
        noised_train = noise(train, target_columns, intensity)
        noised_test = noise(test, target_columns, intensity)

      elif noise_type == 'missing':
        missing_ratio = parameter['missing_ratio']
        target_columns = random_select_columns(int(len(train.columns) * feature_ratio), train.columns)
        noised_train = missing(train, target_columns, missing_ratio)
        noised_test = missing(test, target_columns, missing_ratio)
      
      elif noise_type == "duplicate":
        duplicate_ratio = parameter['duplicate_ratio']
        duplicate_length = parameter['duplicate_length']
        target_columns = random_select_columns(int(len(train.columns) * feature_ratio), train.columns)
        noised_train = duplicate(train, target_columns, duplicate_ratio, duplicate_length)
        noised_test = duplicate(test, target_columns, duplicate_ratio, duplicate_length)

      elif noise_type == "delay":
        delay_length = parameter['delay_length']
        target_columns = random_select_columns(int(len(train.columns) * feature_ratio), train.columns)
        noised_train = delay(train, target_columns, delay_length)
        noised_test = delay(test, target_columns, delay_length)

      # Save the modified dataset
      save_dataset(noised_train, noised_test, labels, dataset, noise_type, level, config['output_folder'], config['validate'], config['validation_ratio'])
  elif noise_type.startswith('mix'):
    types_list = config['noise_types'][noise_type]['types']
    train_original = train.copy()
    test_original = test.copy()
    
    # Keep the original dataset for each noise type, so the high level will not mix with the low level

    for level in ['low', 'high']:
      print(f"mixing noise type: {noise_type}, level: {level}, consist of {types_list}")
      train = train_original.copy()
      test = test_original.copy()
      for type_name in types_list:
        # search for the item in the list with level attribute matching level_name
        parameter = config['noise_types'][type_name][level]
        feature_ratio = parameter['feature_ratio']
        
        if type_name == 'noise':
          # gaussian noise need to be added to numerical columns
          intensity = parameter['intensity']
          numeric_columns = [col for col in train.columns if col not in cate]
          target_columns = random_select_columns(int(len(numeric_columns) * feature_ratio), numeric_columns)
          train = noise(train, target_columns, intensity)
          test = noise(test, target_columns, intensity)
        elif type_name == 'missing':
          missing_ratio = parameter['missing_ratio']
          target_columns = random_select_columns(int(len(train.columns) * feature_ratio), train.columns)
          train = missing(train, target_columns, missing_ratio)
          test = missing(test, target_columns, missing_ratio)
        elif type_name == "duplicate":
          duplicate_ratio = parameter['duplicate_ratio']
          duplicate_length = parameter['duplicate_length']
          target_columns = random_select_columns(int(len(train.columns) * feature_ratio), train.columns)
          train = duplicate(train, target_columns, duplicate_ratio, duplicate_length)
          test = duplicate(test, target_columns, duplicate_ratio, duplicate_length)
        elif type_name == "delay":
          delay_length = parameter['delay_length']
          target_columns = random_select_columns(int(len(train.columns) * feature_ratio), train.columns)
          train = delay(train, target_columns, delay_length)
          test = delay(test, target_columns, delay_length)

      save_dataset(train, test, labels, dataset, noise_type, level, config['output_folder'], config['validate'], config['validation_ratio'])


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
  for noise_type in types:
    print(f"Generating noise data for {noise_type}")
    # Generate noise data for the specified type
    generate_noise(dataset, train, test, labels, categorical_column, config, noise_type)