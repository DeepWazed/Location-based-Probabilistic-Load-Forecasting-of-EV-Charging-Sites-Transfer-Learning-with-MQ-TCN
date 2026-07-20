import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, feature_size):
        super(Attention, self).__init__()
        self.attention_layer = nn.Linear(feature_size, 1)

    def forward(self, features):
        # features shape: [batch_size, num_features, sequence_length]
        attention_scores = self.attention_layer(features.transpose(1, 2))  # [batch_size, sequence_length, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, sequence_length, 1]
        weighted_sum = torch.sum(features * attention_weights.transpose(1, 2), dim=2)  # [batch_size, num_features]
        return weighted_sum, attention_weights

class QuantileTaskTCN(nn.Module):
    def __init__(self, input_size, embedding_dims, num_channels, kernel_size, dropout, output_size, quantiles):
        super(QuantileTaskTCN, self).__init__()
        self.quantiles = quantiles
        self.embedding = nn.ModuleList([nn.Embedding(num_categories, emb_dim) for num_categories, emb_dim in embedding_dims])
        
        self.tcn = nn.Sequential(
            *[TemporalBlock(num_channels[i-1] if i > 0 else input_size + sum([emb_dim for _, emb_dim in embedding_dims]),
                            num_channels[i],
                            kernel_size=kernel_size,
                            stride=1,
                            dilation=2 ** i,
                            padding=(kernel_size - 1) * 2 ** i,
                            dropout=dropout) for i in range(len(num_channels))]
        )
        self.attention = Attention(num_channels[-1])
        self.fc = nn.ModuleList([nn.Linear(num_channels[-1], output_size) for _ in quantiles])

    def forward(self, x_num, x_cat):
        x_cat_embedded = [emb(x_cat[:, :, i]) for i, emb in enumerate(self.embedding)]
        x_cat_embedded = torch.cat(x_cat_embedded, dim=-1)
        x = torch.cat((x_num, x_cat_embedded), dim=-1)
        x = x.permute(0, 2, 1)
        y = self.tcn(x)

        y_att, _ = self.attention(y)
        # y_att = y_att.to(DEVICE)

        y_list = [self.fc[i](y_att) for i in range(len(self.quantiles))] 
        y_pred = torch.stack(y_list, dim=-1)
        return y_pred