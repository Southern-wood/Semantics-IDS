import pickle
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.constants import device, args, color
from src.model.our import FeatureProxy
from src.utils.metrics import pot_eval, adjust_predicts
from src.utils import *
from src.utils.generate_testfiles import *
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import torch.nn as nn
import torch.nn.functional as F
from time import time

def convert_to_windows(data, model):
  windows = []
  w_size = model.n_window
  print(data.shape)
  for i, g in enumerate(data): 
    if i >= w_size:
      w = data[i - w_size + 1 : i + 1]
    # This may lead to inaccurate window data
    else:
      w = torch.cat([data[0].repeat(w_size - i - 1, 1), data[0 : i + 1]])
    windows.append(w)
  return torch.stack(windows)

def convert_labels_to_windows(labels, w_size):
  windowed_labels = []
  n = len(labels)
  for i in range(n):
    if i >= w_size -1 : 
        window_slice = labels[i - w_size + 1 : i + 1]
    else:
        window_slice = labels[0 : i + 1]

    # Label is 1 if any label in the slice is 1
    windowed_labels.append(np.max(window_slice)) # np.max works for 0/1 labels
  return np.array(windowed_labels)

def save_model_self(model, model_save_path, optimizer, scheduler, epoch, accuracy_list):    
  if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
  if not os.path.exists(os.path.join(model_save_path, str(args.dataset))):
    os.makedirs(os.path.join(model_save_path, str(args.dataset)))
  torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'accuracy_list': accuracy_list}, file_path)


def load_model(modelname, dims, batch_size, file_path):
  import src.model.our
  model_class = getattr(src.model.our, modelname)
  model = model_class(dims, batch_size).double()
  model = model.to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
  if os.path.exists(file_path) and args.mode == 'test':
    print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
    checkpoint = torch.load(file_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    accuracy_list = checkpoint['accuracy_list']
    print("Current Model Trained Epoch:" + str(epoch))
  elif args.mode == 'train':
    print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
    epoch = -1
    accuracy_list = []
  else:
    print(f"{color.RED}Model not found: {file_path}{color.ENDC}")
    exit()
  return model, optimizer, scheduler, epoch, accuracy_list


def backprop(enhanced_model, data, optimizer, scheduler, mode='train'):
  MSELoss = nn.MSELoss(reduction='none')
  dataset = TensorDataset(data, data)
  dataloader = DataLoader(dataset, batch_size=enhanced_model.detector.batch)

  update_freq = enhanced_model.update_freq
  total_loss = 0
  global_step = 0

  loss_list = []
  
  # Precompute empty array for faster percentile calculations
  if mode == 'feature_selection':
    from tdigest import TDigest
    digest = TDigest()
    update_counter = 0
    reliable_batch_step = 0
    mini_size = 0.1 * len(dataloader.dataset)
    updated_index = [] # to store updated indices

  for d, _ in tqdm(dataloader):
    window = d.permute(1, 0, 2)
    elem = window[-1, :, :].view(1, d.shape[0], data.shape[2])

    # window: time_step/window, batch_size, features
    # element: the last window of all data: 1, batch_size, features
    window = window.to(device)
    elem = elem.to(device)
    z = enhanced_model(window, elem, mode)

    # calculate the loss
    loss = MSELoss(z, elem)
    total_loss += loss.sum().item()

    if mode == 'train':
      enhanced_model.detector_optimizer.zero_grad()
      loss.mean().backward()
      enhanced_model.detector_optimizer.step()

    elif mode == 'feature_selection':
      # Update the TDigest with Raw loss（without feature selection)
      with torch.no_grad():
        raw_z = enhanced_model(window, elem, mode='train')
        raw_loss = MSELoss(raw_z, elem)
        raw_loss = raw_loss.detach().cpu().numpy()
        raw_loss = raw_loss.sum(axis=2)
        raw_loss = raw_loss.reshape(-1)
        for i in range(raw_loss.shape[0]):
          digest.update(raw_loss[i])
        update_counter += len(raw_loss)
        # print(f"Update Counter: {update_counter}")

      reliable_threhold = digest.percentile(40)
      mean_loss = np.mean(raw_loss)
      if mean_loss < reliable_threhold and update_counter > mini_size: 
        reliable_batch_step += 1
      else:
        continue
      
      selector_update = (reliable_batch_step % update_freq) == (update_freq - 1)
      # Only calculate percentile and update if we have enough history
      if selector_update:
        enhanced_model.weights_optimizer.zero_grad()
        loss.mean().backward()
        enhanced_model.update_tau(global_step)
        enhanced_model.weights_optimizer.step()
        updated_index.append(global_step)
        # print("Updated index: ", global_step)
    
      loss = loss[0]
      loss_list.append(loss.detach().cpu())
    elif mode == 'test':
      loss = loss[0]
      loss_list.append(loss.detach().cpu())

    global_step += 1
  
  if mode == 'train':
    scheduler.step()
    avg_loss = total_loss / len(dataloader)
    return avg_loss, optimizer.param_groups[0]['lr']
  elif mode == 'feature_selection':
    return updated_index
  elif mode == 'test':
    loss_list = torch.cat(loss_list, 0)
    return loss_list.detach().numpy()

if __name__ == '__main__':
  model_name = "TranAD_TNT_AutoDIS_SelfAtt_LSTM_ASSA_TOP_M"

  dataset = args.dataset
  quality_type = args.quality_type
  level = args.level
  num_epoch = args.num_epoch
  batch_size = args.batch_size
  model_save_path = args.model_save_path

  if quality_type == 'pure':
    level = ''


  file_path = os.path.join(model_save_path, dataset, quality_type + level + '_checkpoint.pth')
  file_path = os.path.abspath(file_path)
  print(f"Model file saved at: {file_path}")
  train_file_path, label_file_path = generate_trainpath_and_label('../processed', dataset, quality_type, level)
  testfiles = generate_testfiles('../processed', dataset)

  train_np = np.load(train_file_path)
  train_data = pd.DataFrame(train_np)

  categorical_column = []
  for entry in train_data.columns:
    if all(np.isclose(round(train_data[entry]), train_data[entry])):
      categorical_column.append(entry)

  train_loader = DataLoader(train_np, batch_size=train_np.shape[0])
  feature_num = train_data.shape[1]


  model, optimizer, scheduler, epoch, accuracy_list = load_model(model_name, feature_num, batch_size, file_path)
  enhanced_model = FeatureProxy(model, optimizer, scheduler, feature_num, device)

  trainD = next(iter(train_loader))
  trainD = convert_to_windows(trainD, model)

  ### Training phase
  if args.mode == 'train':
    print(f'{color.HEADER}Training {model_name} on {args.dataset} with num_epochs : {num_epoch}{color.ENDC}')
    # enhanced_model.detector.train()
    num_epochs = num_epoch
    e = epoch + 1
    start = time()
    for e in list(range(epoch + 1, epoch + num_epochs + 1)):
      print(f'\n{color.BOLD}Epoch {e} training of total {num_epochs} epochs{color.ENDC}')
      lossT, lr = backprop(enhanced_model, trainD, optimizer, scheduler, mode='train')
      accuracy_list.append((lossT, lr))
    print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time() - start) + ' s' + color.ENDC)
    save_model_self(model, model_save_path, optimizer, scheduler, e, accuracy_list)
    exit()

  if quality_type == 'pure':
    test_file_path = next((path for path in testfiles if quality_type in path), None)
  else:
    test_file_path = next((path for path in testfiles if quality_type + '_' + level in path), None)
  print(f"Test file path: {test_file_path}")
  test_np = np.load(test_file_path)
  test_data = pd.DataFrame(test_np)
  labels = np.load(label_file_path)
  test_loader = DataLoader(test_np, batch_size=test_np.shape[0])

  testD = next(iter(test_loader))
  testD = convert_to_windows(testD, model)

  enhanced_model.train()

  enhanced_model.weights.data = torch.zeros(feature_num, device=device)

  ### Testing phase
  labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
  labelsFinal = convert_labels_to_windows(labelsFinal, model.n_window)
  print(f'\n{color.BOLD}Labels shape: {labelsFinal.shape}{color.ENDC}')
  print("anomaly rate: ", labelsFinal.mean())

  # Convert labels to windows
  print(f'{color.HEADER}Testing {model_name} on {args.dataset}{color.ENDC}')
  print(f'\n{color.BOLD}Auto Feature Selection on {test_file_path} {color.ENDC}')
  updated_index = backprop(enhanced_model, testD, optimizer, scheduler, mode='feature_selection')

  ### Print feature selection info: how many updates are reliable 
  print(f"\n{color.BOLD}Feature Selection Info{color.ENDC}")
  enhanced_model.info(updated_index, labelsFinal)

  enhanced_model.eval()

  with torch.no_grad():
    print(f'\n{color.BOLD}Evaluating on {test_file_path} {color.ENDC}')
    loss = backprop(enhanced_model, testD, optimizer, scheduler, mode='test')
    print(f'\n{color.BOLD}Getting loss on Training set for POT {color.ENDC}')
    lossT = backprop(enhanced_model, trainD, optimizer, scheduler, mode='test')


  indices = torch.where(enhanced_model.weights >= 0)[0]

  print(f"\n{color.BOLD}Feature Selection Indices: {indices}\nCount: {len(indices)}{color.ENDC}")
  print(f"\n{color.BOLD}POT for Selected Features")

  df = pd.DataFrame()


  for feature_index in range(lossT.shape[1]):

    if feature_index not in indices:
      continue

    feature_lossT = lossT[:, feature_index]
    feature_loss = loss[:, feature_index]

		# pot_eval return raw prediction, without point adjustment
    _, feature_y_pred = pot_eval(feature_lossT, feature_loss, labelsFinal)
    if _ == {}:
      continue
    print("cate : " if feature_index in categorical_column else "num : ", feature_index, 
          " f1 = " , _['f1'], 
          " precision = ", _['precision'], 
          " recall = ",_['recall']) 
    df[feature_index] = feature_y_pred
  
  positive_sum = df.sum(axis=1)

  best_f1, best_threshold = 0, 0
  np.max(positive_sum)
  for i in range(positive_sum.max() + 1):
    threshold = i
    prediction = adjust_predicts(positive_sum, labelsFinal, threshold)
    f1 = f1_score(labelsFinal, prediction)
    if f1 > best_f1:
      best_f1 = f1
      best_threshold = threshold
    print(f"Threshold: {threshold}, F1 Score: {f1}")

  print(f"Best Threshold: {best_threshold}, Best F1 Score: {best_f1}")