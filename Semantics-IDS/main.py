import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
import pandas as pd
import numpy as np
import os 
from time import time

from src.model.trans_semantics import Trans_Semantics
from src.model.feature_proxy import FeatureProxy
from src.constants import args, color

from src.utils.FSDP_warp import setup, cleanup, fsdp_wrapper_model, create_logger
from src.utils.data_loader import get_loader_segment, get_normalization
from src.utils.metrics import pot_eval, eval_f1score, adjust_predicts
from src.utils.path_handle import dataset_path, model_path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def save_model(model, model_save_path, optimizer, scheduler, epoch, accuracy_list):   
  if dist.get_rank() == 0:
    model_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
  dist.barrier() 

  with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
      model_state = model.state_dict()
      optim_state = optimizer.state_dict()

  if dist.get_rank() == 0:
    torch.save({
      'epoch': epoch,
      'model_state_dict': model_state,
      'optimizer_state_dict': optim_state,
      'scheduler_state_dict': scheduler.state_dict(),
      'accuracy_list': accuracy_list}, model_save_path)
    logger.info(f"Model saved to {model_save_path} at epoch {epoch}")
  dist.barrier() 


def load_model(dims, batch_size, model_save_path):
  model = Trans_Semantics(dims, batch_size)
  model = fsdp_wrapper_model(model)

  optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-5)
  
  epoch = -1
  accuracy_list = []
  if os.path.exists(model_save_path) and args.mode == 'test':
    checkpoint = torch.load(model_save_path, map_location="cpu")
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if dist.get_rank() == 0:
      logger.info(f"{color.GREEN}Loading trained model: {model.name}{color.ENDC}")
      epoch = checkpoint['epoch']
      accuracy_list = checkpoint.get('accuracy_list', [])

    current_device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    loaded_epoch_tensor = torch.tensor(0, dtype=torch.long).to(current_device) 

    if dist.get_rank() == 0:
        loaded_epoch_tensor.fill_(epoch)
    dist.broadcast(loaded_epoch_tensor, src=0)
    if dist.get_rank() != 0:
        epoch = loaded_epoch_tensor.item()
    dist.barrier()


  elif args.mode == 'train':
    if dist.get_rank() == 0:
      logger.info(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
  else:
    if dist.get_rank() == 0:
        logger.error(f"{color.RED}Model not found at {model_save_path} or invalid mode '{args.mode}' for loading.{color.ENDC}")
    dist.barrier()
    raise FileNotFoundError(f"Model not found at {model_save_path} or invalid mode '{args.mode}' for loading.")
  return model, optimizer, scheduler, epoch, accuracy_list

def train_step(enhanced_model, data_loader, normalization):
  MSELoss = nn.MSELoss(reduction='none')
  mean_tensor = torch.from_numpy(normalization.mean_)
  scale_tensor = torch.from_numpy(normalization.scale_)
  total_loss = 0

  time_record = time()
  logging_interval = 100

  enhanced_model.train()
  
  for global_step, data in enumerate(data_loader):
    
    window = data.permute(1, 0, 2)
    window = window[:-1, :, :]
    elem = window[-1, :, :].view(1, data.shape[0], data.shape[2])

    # Shape of window: (n_window, batch_size, n_features)

    z = enhanced_model(window)
    elem = elem.to(device=z.device, dtype=z.dtype)
    z, elem = z.squeeze(0), elem.squeeze(0)
    mean_tensor = mean_tensor.to(device=z.device)
    scale_tensor = scale_tensor.to(device=z.device)
    z = (z - mean_tensor) / scale_tensor
    elem = (elem - mean_tensor) / scale_tensor
    loss = MSELoss(z, elem)
    total_loss += loss.sum().item() / data.shape[0]
    
    enhanced_model.detector_optimizer.zero_grad()
    loss.mean().backward()
    enhanced_model.detector_optimizer.step()

    if global_step % logging_interval == logging_interval - 1 and dist.get_rank() == 0:
      Speed = logging_interval / (time() - time_record) # it/s
      remain_time = (len(data_loader) - global_step) / Speed
      logger.info(f"Speed: {Speed:.2f} it/s, Remaining time: {remain_time:.2f}s")
      time_record = time()
  
    # use in-epoch learning rate scheduler since datasets is too big
    enhanced_model.detector_scheduler.step()
  avg_loss = total_loss / (len(data_loader) * data_loader.batch_size)
  return avg_loss, optimizer.param_groups[0]['lr']


def inference(enhanced_model, data_loader, normalization):
  MSELoss = nn.MSELoss(reduction='none')
  mean_tensor = torch.from_numpy(normalization.mean_)
  scale_tensor = torch.from_numpy(normalization.scale_)
  loss_list = []
  
  enhanced_model.eval()

  time_record = time()
  logging_interval = 100

  for global_step, batch_data in enumerate(data_loader):
    if len(batch_data) == 2:
      data, _ = batch_data  
    else:
      data = batch_data
    # logger.info(f"Batch data shape: {data.shape}")
    window = data.permute(1, 0, 2)
    window = window[:-1, :, :]
    elem = window[-1, :, :].view(1, data.shape[0], data.shape[2])

    z = enhanced_model(window, 'test')
    elem = elem.to(z.device)
    mean_tensor = mean_tensor.to(device=z.device)
    scale_tensor = scale_tensor.to(device=z.device)
    z, elem = z.squeeze(0), elem.squeeze(0)
    z = (z - mean_tensor) / scale_tensor
    elem = (elem - mean_tensor) / scale_tensor
    loss = MSELoss(z, elem)
    loss_list.append(loss.detach().cpu())

    if global_step % logging_interval == logging_interval - 1 and dist.get_rank() == 0:
      Speed = logging_interval / (time() - time_record) # it/s
      remain_time = (len(data_loader) - global_step) / Speed
      logger.info(f"Speed: {Speed:.2f} it/s, Remaining time: {remain_time:.2f}s")
      time_record = time()
  
  loss_list = torch.cat(loss_list, 0)
  return loss_list.detach().numpy()

if __name__ == '__main__':
  # Set resource limits
  setup()
  global logger
  logger = create_logger()
  
  
  dataset = args.dataset
  quality_type = args.quality_type
  level = args.level
  feat_num = args.feat_num
  num_epoch = args.num_epoch
  batch_size = args.batch_size
  model_save_path = args.model_save_path
  prefix = args.data_path
  model_save_path = os.path.abspath(model_path(model_save_path, dataset, quality_type, level))

  logger.info(f"Model file path: {model_save_path}")

  model, optimizer, scheduler, epoch, accuracy_list = load_model(feat_num, batch_size, model_save_path)
  enhanced_model = FeatureProxy(model, optimizer, scheduler, feat_num, args.feature_selection_batch_size, args.relability_rate)

  if args.mode == 'train':
    logger.info(f'{color.HEADER}Training Trans-Semantics on {args.dataset} with num_epochs : {num_epoch}{color.ENDC}')
    enhanced_model.detector.train()
    num_epochs = num_epoch
    e = epoch + 1
    start = time()

    # build train_loader
    train_path = dataset_path(prefix, dataset, quality_type, level, 'train')
    logger.info(f"Train path: {train_path}")
    train_loader = get_loader_segment(mode='train', normal_path=train_path, batch_size=batch_size, win_size=model.n_window)
    norm = get_normalization(normal_path=train_path, win_size=model.n_window)

    for e in list(range(epoch + 1, epoch + num_epochs + 1)):
      logger.info(f'{color.BOLD}Epoch {e} training of total {num_epochs} epochs{color.ENDC}')
      lossT, lr = train_step(enhanced_model, data_loader=train_loader, normalization=norm)
      accuracy_list.append((lossT, lr))
      logger.info(color.BOLD + 'Training time: ' + "{:10.4f}".format(time() - start) + ' s' + color.ENDC)
      logger.info(color.BOLD + 'Training loss: ' + "{:10.4f}".format(lossT) + color.ENDC)
    save_model(model, model_save_path, optimizer, scheduler, e, accuracy_list)
    cleanup()
    exit()

  train_path = dataset_path(prefix, dataset, quality_type, level, 'train')
  norm = get_normalization(normal_path=train_path, win_size=model.n_window)
  test_path, label_path = dataset_path(prefix, dataset, quality_type, level, 'test')
  val_path = dataset_path(prefix, dataset, quality_type, level, 'val')

  # build dataloaders
  train_loader = get_loader_segment(mode='train', noshuffle=True, normal_path=train_path, batch_size=batch_size, win_size=model.n_window)
  val_loader = get_loader_segment(mode='val', normal_path=train_path, validation_path=val_path, batch_size=args.feature_selection_batch_size, win_size=model.n_window)  
  test_loader = get_loader_segment(mode='test', normal_path=train_path, attack_path=test_path, labels_path=label_path, batch_size=batch_size, win_size=model.n_window)

  # Extract labels from the label_loader 
  labelsFinal = []
  for _, label in test_loader:
      labelsFinal.extend(label.cpu().numpy())
  labelsFinal = np.array(labelsFinal, dtype=int)
  
  logger.info("anomaly rate: " + str(np.sum(labelsFinal) / len(labelsFinal)))

  enhanced_model.train()
  enhanced_model.weights.data = torch.zeros(feat_num)

  logger.info(f'{color.HEADER}Adapting Trans-Semantics on {args.dataset}{color.ENDC}')
  logger.info(f'{color.BOLD}Auto Feature Selection on {val_path} {color.ENDC}')
  

  for i in range(args.feature_selection_num_epoch):
    logger.info(f'{color.BOLD}Epoch {i} Adapting of total {args.feature_selection_num_epoch} epochs{color.ENDC}')
    reliable_time_indices = enhanced_model.feature_selection(val_loader, normalization=norm)

  indices = enhanced_model.selected_features()
  logger.info(f"{color.BOLD}Selected Features Count: {len(indices)}{color.ENDC}")

  enhanced_model.eval()

  with torch.no_grad():
    logger.info(f'{color.BOLD}Evaluating on {test_path} {color.ENDC}')
    loss = inference(enhanced_model, data_loader=test_loader, normalization=norm)
    logger.info(f'{color.BOLD}Getting loss on Training set for POT {color.ENDC}')
    lossT = inference(enhanced_model, data_loader=train_loader, normalization=norm)

  indices = enhanced_model.selected_features()

  logger.info(f"{color.BOLD}POT for Selected Features...{color.ENDC}") 

  df = pd.DataFrame()
  for feature_index in range(lossT.shape[1]):
    if feature_index not in indices:
      continue
      
    feature_lossT = lossT[:, feature_index]
    feature_loss = loss[:, feature_index]

    # pot_eval return raw prediction, without point adjustment
    _, feature_y_pred = pot_eval(feature_lossT, feature_loss, labelsFinal)
    prediction_rate = np.sum(feature_y_pred) / len(feature_y_pred)
    if prediction_rate > 0.5: # if the prediction rate makes no sense, we just skip this feature
      continue
    df[feature_index] = feature_y_pred
  
  positive_sum = df.sum(axis=1)

  from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score, f1_score

  f1s, thresholds = eval_f1score(positive_sum, labelsFinal)
  indices = np.where(f1s == np.max(f1s))[0]

  f1 = f1s[indices[0]]
  threshold = thresholds[indices[0]]
  pred = adjust_predicts(positive_sum, labelsFinal, threshold)

  precision = precision_score(labelsFinal, pred)
  recall = recall_score(labelsFinal, pred)
  accuracy = accuracy_score(labelsFinal, pred)
  tn, fp, fn, tp = confusion_matrix(labelsFinal, pred).ravel()
  specificity = tn / (tn + fp)

  # Calculate ROC AUC if possible
  try:
    auc = roc_auc_score(labelsFinal, positive_sum)
  except:
    auc = float('nan')  # In case of only one class being present

  logger.info(f"{color.BOLD}Evaluation Metrics:{color.ENDC}")
  logger.info(f"Precision: {precision:.4f}")
  logger.info(f"Recall: {recall:.4f}")
  logger.info(f"Accuracy: {accuracy:.4f}")
  logger.info(f"Specificity: {specificity:.4f}")
  logger.info(f"AUC: {auc:.4f}")
  logger.info(f"F1 Score: {f1:.4f}")

  cleanup()