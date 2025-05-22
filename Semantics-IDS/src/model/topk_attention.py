import torch
import torch.nn as nn

from timm.layers import DropPath,  trunc_normal_

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
    def __init__(self, dim, win_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.1, proj_drop=0.1, top_m=99):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.top_m = top_m
        
        self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        
        self.attn_drop = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(attn_drop),
        )
        self.proj_drop = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop),
        )

        self.softmax= nn.Softmax(dim=-1)

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

        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        raw_attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(raw_attn)

        mask = torch.zeros(B_, self.num_heads, N, N, device=q.device, requires_grad=False)
        # print("top_m: ", self.top_m, raw_attn.shape[-1])
        topm = min(self.top_m, raw_attn.shape[-1])
        index = torch.topk(raw_attn, k=topm, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn_topk = torch.where(mask>0, raw_attn, torch.full_like(raw_attn, float('-inf')))
        attn_topk = self.softmax(attn_topk)
        
        # attn_sprase = self.relu(raw_attn) ** 2
        
        w0 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w1 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        # print("w1: ", w1, "w2: ", w2)
        # attn = w0 * attn + w1 * attn_topk + w2 * attn_sprase
        attn = w0 * attn + w1 * attn_topk
        attn = self.attn_drop(attn)
        # attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(x) 

        return x

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
    