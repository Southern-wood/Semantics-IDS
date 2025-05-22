import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from ..constants import lr, args
from .topk_attention import TopM_MHSA

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# torch.manual_seed(1)

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
		self.register_buffer('index_tensor', torch.arange(0, self.buck_size))
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


class Trans_Semantics(nn.Module):
	def __init__(self, feats, batch_size):
		super(Trans_Semantics, self).__init__()
		self.name = 'Trans_Semantics'
		self.lr = lr
		self.batch = int(batch_size) 
		self.n_feats = feats  #The total number of features
		self.n_window = 10
		self.input_window = self.n_window - 1
		self.bucket_size = 6
		self.embedding = 6  #Each numerical and categorical features same embedding size
		self.num_heads = 3
		self.num_mhsa_layers = 3
		self.dim_feedforward = 12
		self.hidden_dim = self.n_feats * self.embedding
		self.gru_layers = 1
		
	
		#Used for feature value embedding
		self.embedding_layers = nn.ModuleList()
		for i in range(0,self.n_feats):
			self.embedding_layers.append(AutoDis(self.bucket_size,self.embedding))
		
		#Used for spatial property modeling
	
		self.topm_mhsa = TopM_MHSA(self.embedding, (self.input_window, self.n_feats), self.num_heads, self.num_mhsa_layers, self.dim_feedforward, dropout=0.1, top_m=99)

		# self.local_transformer_encoder = TransformerEncoder(local_encoder_layers, 1)
		
		#Used for temporal property modeling
		self.register_buffer('init_h', torch.randn(self.gru_layers, self.batch, self.embedding * self.n_feats)) 
        
		
		self.gru = nn.GRU(input_size=self.embedding*self.n_feats, hidden_size=self.hidden_dim, num_layers=self.gru_layers, bidirectional=False)

		#Used for changing the embedded features back
		self.fcn = nn.Sequential(nn.Linear(self.n_feats*self.embedding, self.n_feats), nn.Sigmoid())

	def local_encode(self,input):
		input = input * math.sqrt(self.embedding)
		# print(input.shape)
		transformed_input = self.topm_mhsa(input)
		return transformed_input

	def forward(self, src):
		src = src.unsqueeze(-1)
		batch_size = src.shape[1]
		src_chunked = src.chunk(src.shape[2],dim=2)

		assert len(src_chunked) == self.n_feats
		src_chunked_embedded_all = []
		for i in range(0,self.n_feats):
			current_src_chunked = src_chunked[i].squeeze(-1)
			current_src_chunked = torch.flatten(current_src_chunked,start_dim=0,end_dim=1)
			#time_step/window*batch_size,1
			module_device = next(self.embedding_layers[i].parameters()).device
			if current_src_chunked.device != module_device:
					self.embedding_layers[i].to(current_src_chunked.device)
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
		src_embedded = rearrange(src_embedded, 'w b f e -> b (w f) e')
		src_embedded = self.local_encode(src_embedded)
		src_embedded = rearrange(src_embedded, 'b (w f) e -> w b (f e)', w = self.input_window)

		initial_hidden_state = self.init_h[:, :batch_size, :].contiguous()
		gru_out, hidden = self.gru(src_embedded[:-1,:,:], initial_hidden_state)
		#Only return the last output
		final_out = self.fcn(gru_out[-1,:,:]).unsqueeze(0)

		return final_out
