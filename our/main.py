import pickle
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.constants import device, args, color
from src.model.our import FeatureProxy
from src.utils.metrics import pot_eval_our
from src.utils import *
from src.utils.generate_testfiles import *
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from time import time

def convert_to_windows(data, model):
	windows = []
	w_size = model.n_window
	print(data.shape)
	for i, g in enumerate(data): 
		if i >= w_size:
			w = data[i-w_size+1:i+1]
		# This may lead to inaccurate window data
		else:
			w = torch.cat([data[0].repeat(w_size-i-1, 1), data[0:i+1]])
		windows.append(w)
	return torch.stack(windows)


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


def load_model(modelname, dims):
	import src.model.our
	model_class = getattr(src.model.our, modelname)
	model = model_class(dims).double()
	model = model.to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	fname = file_path
	if os.path.exists(fname) and (not args.retrain or args.mode == 'test'):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname, weights_only=False)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
		print("Current Model Trained Epoch:" + str(epoch))
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1
		accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list


def backprop(enhanced_model, data, optimizer, scheduler, mode='train'):
	MSELoss = nn.MSELoss(reduction='none')
	dataset = TensorDataset(data, data)
	dataloader = DataLoader(dataset, batch_size=enhanced_model.detector.batch)

	update_freq = enhanced_model.update_freq
	total_loss = 0
	global_step = 0

	loss_list = []

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
			selector_update = (global_step % update_freq) == (update_freq - 1)
			if selector_update:
				# Update the feature selection weights
				enhanced_model.weights_optimizer.zero_grad()
				loss.mean().backward()
				enhanced_model.update_tau(global_step)
				enhanced_model.weights_optimizer.step()
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
	elif mode == 'test' or mode == 'feature_selection':
		loss_list = torch.cat(loss_list, 0)
		return loss_list.detach().numpy()


if __name__ == '__main__':
	model_name = "TranAD_TNT_AutoDIS_SelfAtt_LSTM_ASSA_TOP_M"

	dataset = args.dataset
	noise_type = args.noise_type
	noise_level = args.noise_level
	num_epoch = args.num_epoch
	model_save_path = args.model_save_path

	file_path = os.path.join(model_save_path, str(dataset), str(noise_type) + '_' + str(noise_level) + '_checkpoint.pth')
	file_path = file_path.replace('checkpoint', 'checkpoint_assa_top')
	train_file_path, label_file_path = generate_trainpath_and_label('../processed', dataset, noise_type, noise_level)
	testfiles = generate_testfiles('../processed', dataset)

	train_np = np.load(train_file_path)
	train_data = pd.DataFrame(train_np)

	categorical_column = []
	for entry in train_data.columns:
		if all(np.isclose(round(train_data[entry]), train_data[entry])):
			categorical_column.append(entry)

	train_loader = DataLoader(train_np, batch_size=train_np.shape[0])
	feature_num = train_data.shape[1]

	model, optimizer, scheduler, epoch, accuracy_list = load_model(model_name, feature_num, dataset, noise_type, noise_level)
	enhanced_model = FeatureProxy(model, optimizer, scheduler, feature_num, device)

	trainD = next(iter(train_loader))
	trainD = convert_to_windows(trainD, model)

	### Training phase
	if args.mode == 'train':
		print(f'{color.HEADER}Training {model_name} on {args.dataset} with num_epochs : {num_epoch}{color.ENDC}')
		
		num_epochs = num_epoch
		e = epoch + 1
		start = time()
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			print(f'\n{color.BOLD}Epoch {e} training{color.ENDC}')
			lossT, lr = backprop(enhanced_model, trainD, optimizer, scheduler, mode='train')
			accuracy_list.append((lossT, lr))
		print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time()-start) + ' s' + color.ENDC)
		save_model_self(model, model_save_path, optimizer, scheduler, e, accuracy_list)
		exit()

	train_type = str(noise_type) + '_' + str(noise_level)
	test_file_path = next((path for path in testfiles if train_type in path), None)

	test_np = np.load(test_file_path)
	test_data = pd.DataFrame(test_np)
	labels = np.load(label_file_path)
	test_loader = DataLoader(test_np, batch_size=test_np.shape[0])

	testD = next(iter(test_loader))
	testD = convert_to_windows(testD, model)

	enhanced_model.train()
	enhanced_model.weights.data = torch.zeros(feature_num, device=device)
	
	### Testing phase
	print(f'{color.HEADER}Testing {model_name} on {args.dataset}{color.ENDC}')
	print(f'\n{color.BOLD}Auto Feature Selection on {test_file_path} {color.ENDC}')
	loss = backprop(enhanced_model, testD, optimizer, scheduler, mode='feature_selection')
	enhanced_model.eval()

	with torch.no_grad():
		loss = backprop(enhanced_model, testD, optimizer, scheduler, mode='test')
		lossT = backprop(enhanced_model, trainD, optimizer, scheduler, mode='test')
	
	### Scores
	df = pd.DataFrame()
	lossTfinal = np.mean(lossT, axis=1)
	lossFinal = np.mean(loss, axis=1)
	labelsFinal = (np.sum(labels, axis=1) >= 1) + 0

	f1_list = []

	indices = torch.where(enhanced_model.weights < 0)[0]
	indices = indices.cpu().numpy()

	for feature_index in range(lossT.shape[1]):
		if feature_index in indices:
			continue

		feature_lossT = lossT[:, feature_index]
		feature_loss = loss[:, feature_index]
		_, feature_y_pred = pot_eval_our(feature_lossT, feature_loss, labelsFinal)
		print(
			"cate : " if feature_index in categorical_column else "num : ",
			feature_index,
			" f1 = ", _['f1'],
			" precision = ", _['precision'],
			" recall = ", _['recall']
		) 
		df[feature_index] = feature_y_pred
	
	positive_sum = df.sum(axis=1)

	test_file = test_file_path.split('.')[-2].split('/')[-1]
	print("Test File: ", test_file)
	test_noise = test_file.split('_')[-2]
	test_degree = test_file.split('_')[-1]
	save_path = generate_save_path('/data/processed', 'TranAD', dataset, noise_type, noise_level, test_noise, test_degree)
	save_path = save_path.replace('energy_saved', 'energy_saved_AutoFS_testonly')
	print("Save Path: ", save_path)
	dirs = '/'.join(save_path.split('/')[:-1])
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	pickle.dump(positive_sum, open(save_path, 'wb'))
	labels_save_path = save_path.split('.')[0] + '_labels.npy'
	pickle.dump(labelsFinal, open(labels_save_path, 'wb'))

	csv_path = save_path.split('.')[0] + '_raw.csv'
	df.to_csv(csv_path)
	print("CSV Save Path: ", csv_path)
