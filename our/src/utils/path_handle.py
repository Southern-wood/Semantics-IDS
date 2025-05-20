import os

def dataset_path(prefix, dataset, noise_type, level=None, mode='train'):
    train_path = prefix + '/' + dataset + '/train/'

    if noise_type == 'pure':
        train_path = train_path + dataset + '_pure.npy'
    else:
        train_path = train_path + dataset + '_' + noise_type + '_' + level + '.npy'
    
    test_path = train_path.replace('train', 'test')
    label_path = test_path.replace('.npy', '_labels.npy')
    validation_path = test_path.replace('.npy', '_val.npy')

    if mode == 'train':
        return train_path
    elif mode == 'test':
        return test_path, label_path
    elif mode == 'val':
        return validation_path

def model_path(model_save_path, dataset, quality_type, level):
  if quality_type == 'pure':
    file_path = os.path.join(model_save_path, dataset, quality_type + '_checkpoint.pth')
  else:
    file_path = os.path.join(model_save_path, dataset, quality_type + '_' + level + '_checkpoint.pth')
  file_path = os.path.abspath(file_path)
  return file_path
