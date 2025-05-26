import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np

class FeatureProxy(torch.nn.Module):
	def __init__(self, model, optimizer, scheduler, feat_num, batch_size, reliable_rate):
			super().__init__()
			self.feat_num = feat_num
			self.tau = 2.5  # beginning temperature

			# trainable weights for feature selection
			self.weights = nn.Parameter(torch.zeros(feat_num), requires_grad=True)
			self.weights_optimizer = torch.optim.Adam([self.weights], lr=1e-5)
			
			# detector
			self.detector = model
			self.detector_optimizer = optimizer
			self.detector_scheduler = scheduler

			self.batch_size = batch_size
			self.reliability_rate = reliable_rate
		
	def selected_features(self):
			indices = torch.where(self.weights >= 0)[0]
			return indices.detach().cpu().numpy()
	
	def forward(self, window, mode = 'train'):
			
			expected_dtype = next(self.detector.parameters()).dtype
			if window.dtype != expected_dtype:
				window = window.to(dtype=expected_dtype)

			if mode == 'train':
					mask = torch.ones_like(self.weights)
			elif mode == 'feature_selection':
					# generate Gumbel-softmax mask
					logits = torch.stack([self.weights, torch.zeros_like(self.weights)], dim=1)
					mask = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)[:, 0]
			elif mode == 'test':						
					mask = torch.zeros_like(self.weights)
					indices = torch.where(self.weights >= 0)[0]
					mask[indices] = 1
				
			mask = mask.view(1, 1, -1)  # to match window shape

			masked_window = window * mask

			used_feats_num = torch.sum(mask).item()
			# forward through the feature proxy
			if mode == 'feature_selection':
					return self.detector(masked_window), used_feats_num
			else:
					return self.detector(masked_window)
	
	
	def feature_selection(self, data_loader, normalization = None):
		from tdigest import TDigest
		
		MSELoss = nn.MSELoss(reduction='none')
		
		expected_dtype = next(self.detector.parameters()).dtype

		if normalization is not None:
			np_dtype = np.float64 if expected_dtype == torch.double else np.float32
			mean_tensor = torch.from_numpy(normalization.mean_.astype(np_dtype)).to(dtype=expected_dtype)
			scale_tensor = torch.from_numpy(normalization.scale_.astype(np_dtype)).to(dtype=expected_dtype)

		reliable_time_indices = []
		digest = TDigest()
		mini_size = 0.1 * len(data_loader)
		
		reliable_count = 0
		# print("Update frequency: ", self.update_freq)
		
		for batch_idx, d in enumerate(data_loader):
			with torch.no_grad():
				d.to(expected_dtype)
				window = d.permute(1, 0, 2)
				window = window[:-1, :, :]
				elem = window[-1, :, :].view(1, d.shape[0], d.shape[2])

				# Calculate the loss for each sample
				raw_z = self(window, mode='train') # Set train mode to get the raw output
				elem = elem.to(raw_z.device)

				if normalization is not None:
					mean_tensor = mean_tensor.to(device=raw_z.device)
					scale_tensor = scale_tensor.to(device=raw_z.device)
					raw_z = (raw_z - mean_tensor) / scale_tensor
					elem = (elem - mean_tensor) / scale_tensor
				
				raw_loss = MSELoss(raw_z, elem)
				loss_sum = raw_loss.detach().cpu().numpy().sum(axis=2).reshape(-1)
				
				for i in range(len(loss_sum)):
					digest.update(loss_sum[i])

				mean_loss = np.mean(loss_sum)
			
			if batch_idx * self.batch_size > mini_size and mean_loss < digest.percentile(self.reliability_rate):
				reliable_count = reliable_count + 1

				# For gradient calculation with reliable samples
				z, used_feats_num = self(window, mode='feature_selection')
				if normalization is not None:
					mean_tensor = mean_tensor.to(device=raw_z.device)
					scale_tensor = scale_tensor.to(device=raw_z.device)
					z = (z - mean_tensor) / scale_tensor
					elem = (elem - mean_tensor) / scale_tensor
				
				loss = MSELoss(z, elem) * self.feat_num / used_feats_num

				self.weights_optimizer.zero_grad()
				loss.mean().backward()
				self.update_tau(batch_idx)
				self.weights_optimizer.step()
				
				# Add the corresponding real time indices to the list
				reliable_time_indices.append(batch_idx)

		return reliable_time_indices
	
	def update_tau(self, step, decay=0.005):
			self.tau = max(0.1, 2.5 - decay * step)

