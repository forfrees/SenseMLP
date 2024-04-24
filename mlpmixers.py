
import torch
import torch.nn as nn
import math


# 位置编码
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # 这个为什么要增加第一维？
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

# Token编码
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding='same', padding_mode='circular')

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

# # 数据编码
class FNetEmbedding(nn.Module):
    def __init__(self, enc_in, d_model):
        super(FNetEmbedding, self).__init__()
        
        self.token = TokenEmbedding(c_in=enc_in, d_model=d_model)
        self.pos = PositionalEmbedding(d_model=d_model)

    def forward(self, x):
        
        x = self.token(x)+self.pos(x)
        return x


class MlpBlock(nn.Module):

    def __init__(self, hidden_dim, mlp_dim):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.dense_1 = nn.Linear(hidden_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.dense_2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.gelu(x)
        x = self.dense_2(x)
        return x

class MixerBlock(nn.Module):
    def __init__(self, hidden_dim, token_dim, token_mlp_dim, channel_mlp_dim):
        super().__init__()
        self.mlp_token = MlpBlock(token_dim, token_mlp_dim)
        self.mlp_channel = MlpBlock(hidden_dim, channel_mlp_dim)
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        y = self.layer_norm_1(x)
        y = x.permute(0, 2, 1)
        y = self.mlp_token(y)
        y = y.permute(0, 2, 1)
        x = x + y
        y = self.layer_norm_2(x)
        return x + self.mlp_channel(y)



class MlpMixer(nn.Module):

    def __init__(self, token_dim, c_in, hidden_dim, mlp_token_dim, mlp_channel_dim, num_block, num_class):
        super().__init__()
        self.embed = FNetEmbedding(enc_in=c_in, d_model=hidden_dim)
        self.blocks = nn.ModuleList([MixerBlock(hidden_dim, token_dim, mlp_token_dim, mlp_channel_dim) for _ in range(num_block)])
        self.head_layer_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        x = self.embed(x)
        for l in self.blocks:
            x = l(x)
        x = self.head_layer_norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

