# Codes are partly combined from https://github.com/locuslab/TCN
#   and https://github.com/sagelywizard/snail

import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm


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
        # padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=1, padding_mode='circular')

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

# 数据编码
class FNetEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FNetEmbedding, self).__init__()
        # c_in 输入的channel
        self.token = TokenEmbedding(c_in=c_in, d_model = d_model)
        self.pos = PositionalEmbedding(d_model = d_model)

    def forward(self, x):
        x = self.token(x) + self.pos(x)
        return x

class Chomp1d(nn.Module):
  def __init__(self, chomp_size):
    super(Chomp1d, self).__init__()
    self.chomp_size = chomp_size

  def forward(self, x):
    return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
  def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
    super(TemporalBlock, self).__init__()
    self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                       stride=stride, padding=padding, dilation=dilation))
    self.chomp1 = Chomp1d(padding)
    self.relu1 = nn.ReLU()
    self.dropout1 = nn.Dropout2d(dropout)

    self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                       stride=stride, padding=padding, dilation=dilation))
    self.chomp2 = Chomp1d(padding)
    self.relu2 = nn.ReLU()
    self.dropout2 = nn.Dropout2d(dropout)

    self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                             self.conv2, self.chomp2, self.relu2, self.dropout2)
    self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
    self.relu = nn.ReLU()
    self.init_weights()

  def init_weights(self):
    #self.conv1.weight.data.normal_(0, 0.01)
    nn.init.xavier_uniform(self.conv1.weight, gain=np.sqrt(2))
    #self.conv2.weight.data.normal_(0, 0.01)
    nn.init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2))
    if self.downsample is not None:
        #self.downsample.weight.data.normal_(0, 0.01)
        nn.init.xavier_uniform(self.downsample.weight, gain=np.sqrt(2))

  def forward(self, x):
    net = self.net(x)
    res = x if self.downsample is None else self.downsample(x)
    return self.relu(net + res)




class AttentionBlock(nn.Module):
  """An attention mechanism similar to Vaswani et al (2017)
  The input of the AttentionBlock is `BxTxD` where `B` is the input
  minibatch size, `T` is the length of the sequence `D` is the dimensions of
  each feature.
  The output of the AttentionBlock is `BxTx(D+V)` where `V` is the size of the
  attention values.
  Arguments:
      dims (int): the number of dimensions (or channels) of each element in
          the input sequence
      k_size (int): the size of the attention keys
      v_size (int): the size of the attention values
      seq_len (int): the length of the input and output sequences
  """
  def __init__(self, dims, k_size, v_size, seq_len=None):
    super(AttentionBlock, self).__init__()
    self.key_layer = nn.Linear(dims, k_size)
    self.query_layer = nn.Linear(dims, k_size)
    self.value_layer = nn.Linear(dims, v_size)
    self.sqrt_k = math.sqrt(k_size)

  def forward(self, minibatch):
    keys = self.key_layer(minibatch)
    queries = self.query_layer(minibatch)
    values = self.value_layer(minibatch)
    logits = torch.bmm(queries, keys.transpose(2,1))
    # Use numpy triu because you can't do 3D triu with PyTorch
    # TODO: using float32 here might break for non FloatTensor inputs.
    # Should update this later to use numpy/PyTorch types of the input.
    mask = np.triu(np.ones(logits.size()), k=1).astype('uint8')
    mask = torch.from_numpy(mask)
    # do masked_fill_ on data rather than Variable because PyTorch doesn't
    # support masked_fill_ w/-inf directly on Variables for some reason.
    logits.data.masked_fill_(mask, float('-inf'))
    probs = F.softmax(logits, dim=1) / self.sqrt_k
    read = torch.bmm(probs, values)
    return minibatch + read

class TemporalConvNet(nn.Module):
  def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2, max_length=200, attention=False):
    super(TemporalConvNet, self).__init__()
    layers = []
    num_levels = len(num_channels)
    for i in range(num_levels):
      dilation_size = 2 ** i
      in_channels = num_inputs if i == 0 else num_channels[i-1]
      out_channels = num_channels[i]
      layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                               padding=(kernel_size-1) * dilation_size, dropout=dropout)]
      if attention == True:
        layers += [AttentionBlock(dims=max_length, k_size=128, v_size=max_length)]

    self.network = nn.Sequential(*layers)

  def forward(self, x):
    return self.network(x)


class CATblock(nn.Module):
    def __init__(self, channel, c_para):
        super(CATblock, self).__init__()
        self.averpooling = nn.AdaptiveAvgPool1d(1)
        self.maxpooling = nn.AdaptiveMaxPool1d(1)
        hidden_channel = int(channel / c_para)
        self.downLinear = nn.Linear(channel, hidden_channel)
        self.upLinear = nn.Linear(hidden_channel, channel)

    def forward(self, x):  #输入维度： B H L
        cAverWeights = self.averpooling(x).squeeze(-1)
        cAverWeights = self.downLinear(cAverWeights)
        cAverWeights = F.relu(cAverWeights)
        cAverWeights = self.upLinear(cAverWeights).unsqueeze(-1)
        cAverWeights = torch.sigmoid(cAverWeights)

        cMaxWeights = self.maxpooling(x).squeeze(-1)
        cMaxWeights = self.downLinear(cMaxWeights)
        cMaxWeights = F.relu(cMaxWeights)
        cMaxWeights = self.upLinear(cMaxWeights).unsqueeze(-1)
        cMaxWeights = torch.sigmoid(cMaxWeights)

        out = (cAverWeights+cMaxWeights) * x
        return out + x


class TATblock(nn.Module):
    def __init__(self, length, l_para):
        super(TATblock, self).__init__()
        self.averPooling = nn.AdaptiveAvgPool1d(1)
        self.maxPooling = nn.AdaptiveMaxPool1d(1)
        hidden_length = int(length / l_para)
        self.downLinear = nn.Linear(length, hidden_length)
        self.upLinear = nn.Linear(hidden_length, length)
        self.conv1d = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, padding='same')
        # self.relu = F.relu()
        # self.sigmoid = F.sigmoid()

    def forward(self, x): # 输入的维度是 B L H
        lAverWeights = self.averPooling(x).squeeze(-1)
        lAverWeights = self.downLinear(lAverWeights)
        lAverWeights = F.relu(lAverWeights)
        lAverWeights = self.upLinear(lAverWeights).unsqueeze(-1)
        lAverWeights = torch.sigmoid(lAverWeights)

        lMaxWeights = self.maxPooling(x).squeeze(-1)
        lMaxWeights = self.downLinear(lMaxWeights)
        lMaxWeights = F.relu(lMaxWeights)
        lMaxWeights = self.upLinear(lMaxWeights).unsqueeze(-1)
        lMaxWeights = torch.sigmoid(lMaxWeights)

        weights = torch.cat([lAverWeights, lMaxWeights], dim=-1)
        weights = self.conv1d(weights.transpose(2, 1)).transpose(2, 1)

        out = weights * x
        return out + x

class TNT(nn.Module):
  def __init__(self, c_in: int, max_length: int, classes :int,
               channel_size=128, level=3, attention=False):
    super(TNT, self).__init__()

    self.embedd = FNetEmbedding(c_in=c_in, d_model=channel_size)
    self.channels = [channel_size] * level
    self.tcn = TemporalConvNet(channel_size, self.channels, kernel_size = 3, dropout = 0.2, max_length = max_length, attention = attention)

    # model T
    self.fc1 = nn.Linear(channel_size, classes)
    self.init_weights()

  def init_weights(self):
    self.fc1.bias.data.fill_(0)
    nn.init.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))

  def forward(self, inputs):
    embeds = self.embedd(inputs)
    output = self.tcn(embeds.transpose(1,2)).transpose(1,2)
    net =self.fc1(output[:, -1,:]).squeeze()
    return F.log_softmax(net, dim=-1)