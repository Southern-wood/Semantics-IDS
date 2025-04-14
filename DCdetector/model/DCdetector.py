import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .attn import DAC_structure, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding
from .RevIN import RevIN
import numpy as np
# from tkinter import _flatten

from collections import deque
from .consts import *

# device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

def flatten(nested_list):
    result = []
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item))  # 递归调用
        else:
            result.append(item)
    return result



Output = True

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=None):
        # global device
        series_list = []
        prior_list = []
        for attn_layer in self.attn_layers:
            # attn_layer = attn_layer.to(device)
            series, prior = attn_layer(x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
        return series_list, prior_list



class DCdetector(nn.Module):
    def __init__(self, win_size, enc_in, c_out, n_heads=1, d_model=256, e_layers=3, patch_size=[3,5,7], channel=55, d_ff=512, dropout=0.0, activation='gelu', output_attention=True):
        super(DCdetector, self).__init__()
        self.output_attention = output_attention
        self.patch_size = patch_size
        self.channel = channel
        self.win_size = win_size
        
        self.embedding_patch_size = nn.ModuleList()
        self.embedding_patch_num = nn.ModuleList()
        for i, patchsize in enumerate(self.patch_size):
            self.embedding_patch_size.append(DataEmbedding(patchsize, d_model, dropout))
            self.embedding_patch_num.append(DataEmbedding(self.win_size//patchsize, d_model, dropout))

        self.embedding_window_size = DataEmbedding(enc_in, d_model, dropout)
         
        # Dual Attention Encoder
        self.encoder = Encoder(
            [
                AttentionLayer(
                    DAC_structure(win_size, patch_size, channel, False, attention_dropout=dropout, output_attention=output_attention),
                    d_model, patch_size, channel, n_heads, win_size)for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)


    def forward(self, x):
        B, L, M = x.shape #Batch win_size channel
        # log(f"Input shape: {x.shape}")
        # log("Batch size: " + str(B))
        # log("Window size: " + str(L))
        # log("Feature size: " + str(M))
        # log("----------------------------")

        series_patch_mean = []
        prior_patch_mean = []
        revin_layer = RevIN(num_features=M)

        # Instance Normalization Operation
        x = revin_layer(x, 'norm')
        x_ori = self.embedding_window_size(x)

        # log(f"shape passed RevIn: {x_ori.shape}")

        # log("----------------------------")
        
        # Mutil-scale Patching Operation 

        for patch_index, patchsize in enumerate(self.patch_size):
            # log(f"Now patch size: {patchsize}")
            x_patch_size, x_patch_num = x, x
            # log(f"shape before rearrange: {x_patch_size.shape}")

            x_patch_size = rearrange(x_patch_size, 'b l m -> b m l') #Batch channel win_size
            x_patch_num = rearrange(x_patch_num, 'b l m -> b m l') #Batch channel win_size
            
            x_patch_size = rearrange(x_patch_size, 'b m (n p) -> (b m) n p', p = patchsize) 
            x_patch_size = self.embedding_patch_size[patch_index](x_patch_size)
            # log(f"x_patch_size type :  {type(x_patch_size)} and shape : {x_patch_size.shape}")
            x_patch_num = rearrange(x_patch_num, 'b m (p n) -> (b m) p n', p = patchsize) 
            x_patch_num = self.embedding_patch_num[patch_index](x_patch_num)
            # log(f"x_patch_num type :  {type(x_patch_num)} and shape : {x_patch_num.shape}")
            

            series, prior = self.encoder(x_patch_size, x_patch_num, x_ori, patch_index)
            # log(f"series type :  {type(series)} and shape : {series[0].shape}")
            # log(f"prior type :  {type(prior)} and shape : {prior[0].shape}")
            series_patch_mean.append(series), prior_patch_mean.append(prior)
            # series : list of [batch, win_size, win_size]
        series_patch_mean = list(flatten(series_patch_mean))
        prior_patch_mean = list(flatten(prior_patch_mean))
        # log(f"output series type :  {type(series_patch_mean)} and shape : {series_patch_mean[0].shape}")

        # start_time = time.time()
        # series_patch_mean = [item for sublist in series_patch_mean for item in (sublist if isinstance(sublist, (list, tuple)) else [sublist])]
        # series_patch_mean = np.array(series_patch_mean)
        # series_patch_mean = series_patch_mean.flatten()

        # prior_patch_mean = [item for sublist in prior_patch_mean for item in (sublist if isinstance(sublist, (list, tuple)) else [sublist])]
        # prior_patch_mean = np.array(prior_patch_mean)
        # prior_patch_mean = prior_patch_mean.flatten()
        # print(f"Time taken: {end_time - start_time} seconds")
        Output = False
        if self.output_attention:
            return series_patch_mean, prior_patch_mean
        else:
            return None
        

