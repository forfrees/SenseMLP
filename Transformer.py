import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=1, padding='same', padding_mode='circular')

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

# # 数据编码
class FNetEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FNetEmbedding, self).__init__()
        
        self.token = TokenEmbedding(c_in, d_model)
        self.pos = PositionalEmbedding(d_model)

    def forward(self, x):
        
        x = self.token(x) + self.pos(x)
        return x


class AttentionEncoder(nn.Module):
    def __init__(self, d_model, num_layers):
        super(AttentionEncoder, self).__init__()
        self.atteEncoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model,
                                       nhead=4, 
                                       dim_feedforward=d_model*4,
                                       batch_first=True, layer_norm_eps=1e-6,dropout=0.2),
            num_layers=num_layers)
    def forward(self, x):
        return self.atteEncoder(x)
    

class TransModel(nn.Module):
    def __init__(self, c_in, d_model, num_layers, num_classes):
        super(TransModel, self).__init__()
        self.embed = FNetEmbedding(c_in, d_model)
        self.attentionEncoder = AttentionEncoder(d_model, num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        embedding = self.embed(x)
        embedding = self.attentionEncoder(embedding)
        embedding = embedding.mean(dim=1)
        return self.head(embedding)
