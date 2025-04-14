import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from ..constants import lr, device
from .sparse_topk_attention import Saprse_TopM_MHSA
import numpy as np

torch.manual_seed(1)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    #Need to add is_causal=False, otherwise error
    def forward(self, src, src_mask=None, is_causal=False, src_key_padding_mask=None):
        #Reutrn weighted src the same shape
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        #Return the same shape to src
        return src



class AutoDis(nn.Module):
	def __init__(self,bs,dim):
		super(AutoDis, self).__init__()
		self.buck_size = bs
		self.embedding_dim = dim
		self.w1 = nn.Sequential(nn.Linear(1,self.buck_size),nn.LeakyReLU())
		self.w2 = nn.Linear(self.buck_size,self.buck_size)
		self.sf = nn.Softmax(dim=-1)
		self.meta_embeddings = nn.EmbeddingBag(self.buck_size,self.embedding_dim,mode='sum')
		self.index_tensor = torch.IntTensor([value for value in range(0,self.buck_size)]).to(device)
		self.control_factor = 0.5
		self.temperature = 1e-5

	def forward(self,src):
		#Step 1
		w1_result = self.w1(src)
		x = self.sf((self.w2(w1_result) + self.control_factor*w1_result)/self.temperature)
		#Step 2
		current_index_tensor = self.index_tensor.repeat(src.shape[0],1)
		#Step 3
		return self.meta_embeddings(current_index_tensor,per_sample_weights=x)


class TranAD_TNT_AutoDIS_SelfAtt_LSTM_ASSA_TOP_M(nn.Module):
	def __init__(self, feats, batch_size):
		super(TranAD_TNT_AutoDIS_SelfAtt_LSTM_ASSA_TOP_M, self).__init__()
		self.name = 'TranAD_TNT_AutoDIS_SelfAtt_LSTM_ASSA_TOP_M'
		self.lr = lr
		self.batch = int(batch_size) 
		self.n_feats = feats  #The total number of features
		self.n_window = 10
		self.input_window = self.n_window - 1
		self.bucket_size = 6
		self.embedding = 6  #Each numerical and categorical features same embedding size
		self.num_heads = 3
		self.num_mhsa_layers = 1
		self.dim_feedforward = 12
		self.hidden_dim = self.n_feats * self.embedding
		self.lstm_layers = 1
		
	
		#Used for feature value embedding
		self.embedding_layers = nn.ModuleList()
		for i in range(0,self.n_feats):
			self.embedding_layers.append(AutoDis(self.bucket_size,self.embedding))
		
		#Used for spatial property modeling
		#The input shape is [self.n_feats,batch_size,self.embedding]
		# local_encoder_layers = TransformerEncoderLayer(self.embedding, nhead=self.embedding, dim_feedforward=12, dropout=0.1)
		self.ast = Saprse_TopM_MHSA(self.embedding, (self.input_window, self.n_feats), self.num_heads, self.num_mhsa_layers, self.dim_feedforward, dropout=0.1, top_m=80)
		# self.topm_mhsa =  TopM_MHSA(self.embedding, self.num_heads, self.num_mhsa_layers, self.dim_feedforward, dropout=0.1, top_m=99)

		# self.local_transformer_encoder = TransformerEncoder(local_encoder_layers, 1)
		
		#Used for temporal property modeling
		self.init_hidden = (
				#randn
        	torch.randn(self.lstm_layers, self.batch, self.embedding * self.n_feats, dtype=torch.float64).to(device)#,
			)
		
		self.lstm = nn.GRU(input_size=self.embedding*self.n_feats, hidden_size=self.hidden_dim, num_layers=self.lstm_layers, bidirectional=False)

		#Used for changing the embedded features back
		self.fcn = nn.Sequential(nn.Linear(self.n_feats*self.embedding, self.n_feats), nn.Sigmoid())

	def local_encode(self,input):
		input = input * math.sqrt(self.embedding)
		# print(input.shape)
		transformed_input = self.ast(input)
		return transformed_input

	def forward(self, src, tgt):
		#Embedding each features individually
		#Src and Tgt input shape is time_step/window, batch_size, features
		#First unsqueeze the last dimension for embedding
		src = src[:-1]
		src = src.unsqueeze(-1)#.float()
		#Second chunk based on each feature
		batch_size = src.shape[1]
		src_chunked = src.chunk(src.shape[2],dim=2)
		#Third each feature dimension embedding into via an individual embedding layer
		# print(src_chunked)
		# print(len(src_chunked))
		# print(self.n_feats)
		assert len(src_chunked) == self.n_feats
		src_chunked_embedded_all = []
		for i in range(0,self.n_feats):
			# print("debug : ",src_chunked[i].shape)
			current_src_chunked = src_chunked[i].squeeze(-1)
			# print("debug : ",current_src_chunked.shape)
			current_src_chunked = torch.flatten(current_src_chunked,start_dim=0,end_dim=1)
			# print("debug : ",current_src_chunked.shape)
			#time_step/window*batch_size,1
			src_chunked_embedded = self.embedding_layers[i](current_src_chunked)
			#time_step/window*batch_size,d then Change Back
			src_chunked_embedded = src_chunked_embedded.view(self.input_window,batch_size,1,self.embedding)
			# src_chunked_embedded = src_chunked_embedded.view(self.n_window,batch_size,1,self.embedding)
			src_chunked_embedded_all.append(src_chunked_embedded)

		#Concat the embedded results
		src_embedded = torch.cat(src_chunked_embedded_all,dim = 2)#.double()
		#Then, apply the local transformer to capture spatial correlations
		#Now the src shape is time_step/window, batch_size, features, embedding
		#The input/output of Transformer (no batch first) is L,B,D, correspond to features, time_step/window*batch_size, embedding
		
		#Flatten 0 1 dimensions and premute
		# print(src_embedded.shape)
		# print(src_embedded.shape)
		src_embedded = rearrange(src_embedded, 'w b f e -> b (w f) e')
		# bacth_size, (time_step/window, features), embedding
		# print(src_embedded.shape)

		# print("\n src_embedded : ", src_embedded.shape)

		src_embedded = self.local_encode(src_embedded)
		#Then, apply the temporal transformer to the src
		#The input of this Transformer should be time_step/window, batch_size, features*embedding
		src_embedded = rearrange(src_embedded, 'b (w f) e -> w b (f e)', w = self.input_window)

		#Permute, view and then flatten the last dimension
		# src_embedded = src_embedded.permute(1,0,2)
		# # print("debug : ",src_chunked_embedded.shape)
		# # print("debug : {} {} {} {}".format(self.input_window,batch_size,1,self.embedding))
		# src_embedded = src_embedded.view(self.input_window,batch_size,self.n_feats,-1)
		# # src_embedded = src_embedded.view(self.n_window,batch_size,self.n_feats,-1)
		# src_embedded = torch.flatten(src_embedded,start_dim=2,end_dim=3)

		#Only use the first |time_step/window|-1 windows as input, 
		#(self.init_hidden[0][:,:batch_size,:],self.init_hidden[1][:,:batch_size,:])
		lstm_out, hidden = self.lstm(src_embedded[:-1,:,:], self.init_hidden[:,:batch_size,:])
		#Only return the last output
		final_out = self.fcn(lstm_out[-1,:,:]).unsqueeze(0)

		return final_out

class FeatureProxy(torch.nn.Module):
	def __init__(self, model, optimizer, scheduler, feat_num, device):
			super().__init__()
			self.device = device
			self.feat_num = feat_num
			self.tau = 2.5  # beginning temperature

			# trainable weights for feature selection
			self.weights = nn.Parameter(torch.zeros(feat_num, device=device), requires_grad=True)
			self.weights_optimizer = torch.optim.Adam([self.weights], lr=0.01)
			
			# detector
			self.detector = model.to(device)
			self.detector_optimizer = optimizer
			self.detector_scheduler = scheduler

			self.update_freq = 5  # update frequency for feature selection
	
	def forward(self, window, elem, mode = 'train'):
			
			if mode == 'train':
					mask = torch.ones_like(self.weights, device=self.device)
			elif mode == 'feature_selection':
					# generate Gumbel-softmax mask
					logits = torch.stack([self.weights, torch.zeros_like(self.weights)], dim=1)
					mask = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)[:, 0]
			elif mode == 'test':
					indices = torch.where(self.weights < 0)[0]
					mask = torch.ones_like(self.weights, device=self.device)
					mask[indices] = 0
				
			mask = mask.view(1, 1, -1)  # to match window shape

			# apply mask to window and element
			masked_window = window * mask
			masked_elem = elem * mask

			# forward through the feature proxy
			return self.detector(masked_window, masked_elem)
	
	def info(self, updated_index, labelsFinal):
			updated_index = np.array(updated_index)
			batch_size = self.detector.batch

			# Convert batch indices to sample indices
			sample_updated_indices = []
			for batch_idx in updated_index:
					# Calculate the starting sample index for this batch
					start_idx = batch_idx * batch_size
					# Add all sample indices in this batch
					# Make sure we don't go beyond the dataset length
					end_idx = min(start_idx + batch_size, len(labelsFinal))
					sample_updated_indices.extend(range(start_idx, end_idx))

			sample_updated_indices = np.array(sample_updated_indices)

			# Find indices where labels are anomalies (labelsFinal is True/1)
			anomaly_indices = np.where(labelsFinal == 1)[0]

			# Mark updates that occurred during anomalous periods as unreliable
			unreliable_update = np.intersect1d(sample_updated_indices, anomaly_indices)

			selected_indices = np.where(self.weights.cpu() >= 0)[0]


			# Count unique batches for reporting (to avoid double counting)
			unique_updated_batches = len(np.unique(updated_index))
			print(f"Selected features number: {len(selected_indices)}")
			print(f"Selected rate: {len(selected_indices) / self.feat_num * 100:.2f}%")

			if len(selected_indices) == self.feat_num:
				print("\nAll features are selected")
				print("[This is also expected behavior when the test set comes from the same distribution as the training set, especially on pure.]\n")

			print(f"Total update batches: {unique_updated_batches}")
			print(f"Samples used for updates: {len(sample_updated_indices)}")
			print(f"Used samples account for: {len(sample_updated_indices) / len(labelsFinal) * 100:.2f}%")
			print(f"Unreliable updates (during anomalies): {len(unreliable_update)}")
			print(f"Reliability rate: {(len(sample_updated_indices) - len(unreliable_update)) / len(sample_updated_indices) * 100:.2f}%")
			print(f"Dataset original normal rate: {(1 - labelsFinal.mean()) * 100:.2f}%")
	
	def update_tau(self, step, decay=0.0005):
			self.tau = max(0.5, 2.5 - decay * step)

	