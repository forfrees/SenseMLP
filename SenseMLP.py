import torch
from torch import nn
from scipy import linalg
import torch.nn.functional as F
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
        
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=1, padding='same', padding_mode='circular')

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

# # 数据编码
class FNetEmbedding(nn.Module):
    def __init__(self, config):
        super(FNetEmbedding, self).__init__()
        
        self.token = TokenEmbedding(c_in=config['c_in'], d_model=config['d_model'])
        self.pos = PositionalEmbedding(d_model=config['d_model'])

    def forward(self, x):
        
        x = self.token(x) + self.pos(x)
        return x



class FourierFFT(nn.Module):
    
    def __init__(self):
        super(FourierFFT, self).__init__()

    def forward(self, x):
        return torch.fft.fft(x, dim=-1).real
        # return torch.fft.fft(torch.fft.fft(x.float(), dim=-2), dim=-1).real


class FNetlayerChannel(nn.Module):
    def __init__(self, config):
        super(FNetlayerChannel, self).__init__()
        
        self.mixing_layer_norm = nn.LayerNorm(config['d_model'], eps=config['layer_norm_eps'])
        self.feed_forward = nn.Linear(config['d_model'], config['d_model'] * 4)
        self.output_dense = nn.Linear(config['d_model'] * 4, config['d_model'])
        self.output_layer_norm = nn.LayerNorm(config['d_model'], eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.activation = nn.GELU()

    
    def forward(self, hidden_states):
        # intermediate_output = self.output_layer_norm(hidden_states)
        intermediate_output = self.feed_forward(hidden_states) # 线性层
        intermediate_output = self.activation(intermediate_output)
        output = self.output_dense(intermediate_output)  # 线性层
        output = self.dropout(output+hidden_states)
        output = self.output_layer_norm(output) 
        return output


class FNetlayerTemp(nn.Module):
    def __init__(self, config):
        super(FNetlayerTemp, self).__init__()
        self.mixing_layer_norm = nn.LayerNorm(config['token_lens'], eps=config['layer_norm_eps'])
        self.feed_forward = nn.Linear(config['token_lens'], config['token_lens'] * 4)
        self.output_dense = nn.Linear(config['token_lens'] * 4, config['token_lens'])
        self.output_layer_norm = nn.LayerNorm(config['token_lens'], eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.activation = nn.GELU()

    
    def forward(self, hidden_states):
        # intermediate_output = self.output_layer_norm(hidden_states)
        intermediate_output = self.feed_forward(hidden_states) # 线性层
        intermediate_output = self.activation(intermediate_output)
        output = self.output_dense(intermediate_output)  # 线性层
        output = self.dropout(output+hidden_states)
        output = self.output_layer_norm(output)
        return output



class FNetEncoderChannel(nn.Module):
    def __init__(self, config):
        super(FNetEncoderChannel, self).__init__()
        self.config = config
        self.layerReal = nn.ModuleList([FNetlayerChannel(config) for _ in range(config['num_hidden_layers'])])
        # self.layer_norm = nn.LayerNorm(config['d_model'], eps=config['layer_norm_eps'])
        

    def forward(self, hidden_states):
        # states_Real = self.layer_norm(hidden_states)
        for i, layer_module in enumerate(self.layerReal):
            hidden_states = layer_module(hidden_states)
        return hidden_states



class FNetEncoderTemp(nn.Module):
    def __init__(self, config):
        super(FNetEncoderTemp, self).__init__()
        self.config = config
        self.layerReal = nn.ModuleList([FNetlayerTemp(config) for _ in range(config['num_hidden_layers'])])
        # self.layer_norm = nn.LayerNorm(config['token_lens'], eps=config['layer_norm_eps'])

    def forward(self, hidden_states):
        # states_Real =self.layer_norm(hidden_states)
        for i, layer_module in enumerate(self.layerReal):
            hidden_states = layer_module(hidden_states)
        return hidden_states


class DeFNet(nn.Module):
    def __init__(self, config):
        super(DeFNet, self).__init__()
        self.config = config
        self.embeddings = FNetEmbedding(config)
        self.encoderFNetChannel = FNetEncoderChannel(config)
        self.encoderFNetTemp = FNetEncoderTemp(config)
        self.dense2 = nn.Linear((config['d_model']*3), config['classes'])
        self.fft = FourierFFT()
        self.fftFNet = FNetEncoderChannel(config)

    def forward(self, input):
        
        embedding_output = self.embeddings(input)
        states_Channel = self.encoderFNetChannel(embedding_output)
        states_Temp = self.encoderFNetTemp(embedding_output.transpose(2, 1)).transpose(2, 1)
        states_fft = self.fft(embedding_output)
        states_fft = self.fftFNet(states_fft)
        states_Channel = states_Channel.mean(dim=1)
        states_Temp = states_Temp.mean(dim=1)
        states_fft = states_fft.mean(dim=1)
        head_output = torch.cat([states_Channel, states_Temp, states_fft], dim=-1)
        y = F.log_softmax(self.dense2(head_output), dim=-1)
        return y
