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

class QuantileTCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, quantiles):
        super(QuantileTCNModel, self).__init__()
        self.quantiles = quantiles
        self.tcn = nn.Sequential(
            *[TemporalBlock(input_size if i == 0 else num_channels[i - 1],
                            num_channels[i],
                            kernel_size=kernel_size,
                            stride=1,
                            dilation=2 ** i,
                            padding=(kernel_size - 1) * 2 ** i,
                            dropout=dropout) for i in range(len(num_channels))]
        )
        self.fc = nn.ModuleList([nn.Linear(num_channels[-1], output_size) for _ in quantiles])

    def forward(self, x):
        y = self.tcn(x)
        y = y[:, :, -1] 
        y_list = []
        for i in range(len(self.quantiles)):
            y_q = self.fc[i](y)
            y_list.append(y_q)
        y_pred = torch.stack(y_list, dim=-1)
        return y_pred