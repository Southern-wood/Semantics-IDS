import torch
import torch.nn as nn

from timm.layers import DropPath,  trunc_normal_
from xformers.ops import memory_efficient_attention

class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.1, bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1] 
        return q,k,v

class TopM_MHSA(nn.Module):
    def __init__(self, embed_dim, win_size, num_heads, num_mhsa_layers, dim_feedforward, dropout, top_m):
        super().__init__()
        self.nets = nn.ModuleList([TopM_MHSA_Block(embed_dim, win_size, num_heads, dim_feedforward, dropout, top_m) for _ in range(num_mhsa_layers)])

    def forward(self, x):
        output = x
        for layer in self.nets:
            output = layer(output)
        return output
    
class TopM_MHSA_Block(nn.Module):
    def __init__(self, embed_dim, win_size, nhead, dim_feedforward, dropout, top_m):
        super().__init__()
        drop_path_rate = 0.1
        self.attn = Attention_Sparse_Top_M(embed_dim, win_size, nhead, dropout, top_m)
        self.drop_path = DropPath(drop_path_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=dim_feedforward, act_layer=nn.GELU, drop=0.1)

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_x = self.attn(norm_x)
        x = x + self.drop_path(attn_x)
        norm_x = self.norm2(x)
        mlp_x = self.mlp(norm_x)
        x = x + self.drop_path(mlp_x)
        return x

########### window-based self-attention #############
class Attention_Sparse_Top_M(nn.Module):
    def __init__(self, dim, win_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.1, top_m=99):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.top_m = top_m

        self.attn_dropout_p = attn_drop
        
        self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        
        # self.attn_drop = nn.Sequential(
        #     nn.Softmax(dim=-1),
        #     nn.Dropout(attn_drop),
        # )
        self.proj_drop = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop),
        )

        # self.softmax= nn.Softmax(dim=-1)

        self.relu = nn.ReLU()
        self.w = nn.Parameter(torch.ones(2)) 

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Parameter):
            trunc_normal_(m, std=.02)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape

        q_orig, k_orig, v_orig = self.qkv(x, attn_kv)
        q_scaled = q_orig * self.scale
        raw_attn = (q_scaled @ k_orig.transpose(-2, -1)) # raw_attn shape: (B_, num_heads, N_q, Nkv_actual)

        q_xf = q_orig.permute(0, 2, 1, 3) # (B_, N_q, self.num_heads, D_head)
        k_xf = k_orig.permute(0, 2, 1, 3) # (B_, N_kv, self.num_heads, D_head)
        v_xf = v_orig.permute(0, 2, 1, 3) # (B_, N_kv, self.num_heads, D_head)

        output_dense = memory_efficient_attention(
            q_xf, k_xf, v_xf,
            attn_bias=None, # No explicit bias for dense path
            p=self.attn_dropout_p
        )

        Nkv_actual = raw_attn.shape[-1]
        Nkv_padded = (Nkv_actual + 7) // 8 * 8  # Ensure Nkv_padded is a multiple of 8

        topm_k_val = min(self.top_m, Nkv_actual) # Use Nkv_actual for topk
        _, indices_to_keep = torch.topk(raw_attn, k=topm_k_val, dim=-1, largest=True)

        padded_bias_tensor_shape = list(raw_attn.shape)
        padded_bias_tensor_shape[-1] = Nkv_padded
        
        attn_bias_padded = torch.full(
            padded_bias_tensor_shape,
            float('-inf'),
            dtype=raw_attn.dtype,
            device=raw_attn.device
        )

        gather_src_zeros = torch.zeros_like(indices_to_keep, dtype=raw_attn.dtype, device=raw_attn.device)
        attn_bias_padded.scatter_(dim=-1, index=indices_to_keep, src=gather_src_zeros)

        topk_bias_for_xf = attn_bias_padded[:, :, :, :Nkv_actual]
        
        output_topk = memory_efficient_attention(
            q_xf, k_xf, v_xf,
            attn_bias=topk_bias_for_xf, 
            p=self.attn_dropout_p
        )
        exp_w = torch.exp(self.w)
        sum_exp_w = torch.sum(exp_w)
        w0 = exp_w[0] / sum_exp_w
        w1 = exp_w[1] / sum_exp_w
        
        combined_output = w0 * output_dense + w1 * output_topk

        final_x = self.proj_drop(combined_output.contiguous().view(B_, N, C)) # Ensure output has correct shape for proj_drop

        return final_x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    