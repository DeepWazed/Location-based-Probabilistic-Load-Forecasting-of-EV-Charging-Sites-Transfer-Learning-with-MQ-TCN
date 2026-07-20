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
        

class FineTunedTimeSeriesModel(nn.Module):
    def __init__(self, base_model, freeze_embedding, num_frozen_tcn_layers, input_size, embedding_dims, num_channels, kernel_size, dropout, output_size, quantiles):
        super(FineTunedTimeSeriesModel, self).__init__()
        self.quantiles = quantiles
        self.base_embedding = base_model.embedding

        self.num_adapter = nn.Linear(input_size, 41)  

        self.transfer_embedding = nn.ModuleList([nn.Embedding(num_categories, emb_dim) for num_categories, emb_dim in embedding_dims])

        self.mapping_layers = nn.ModuleList([
            nn.Linear(transfer_emb.embedding_dim, base_emb.embedding_dim)
            for transfer_emb, base_emb in zip(self.transfer_embedding, self.base_embedding)
        ])

        if freeze_embedding:
            for param in self.base_embedding.parameters():
                param.requires_grad = False

        base_model.fc = nn.Identity()
        self.frozen_tcn = nn.Sequential(*list(base_model.tcn.children())[:num_frozen_tcn_layers])
        for param in self.frozen_tcn.parameters():
            param.requires_grad = False

        last_tcn_block = list(base_model.tcn.children())[-1]
        additional_tcn_input_size = last_tcn_block.conv2.out_channels
        self.additional_tcn = nn.Sequential(
            *[TemporalBlock(
                num_channels[i - 1] if i > 0 else additional_tcn_input_size,
                num_channels[i],
                kernel_size=kernel_size,
                stride=1,
                dilation=2 ** i,
                padding=(kernel_size - 1) * 2 ** i,
                dropout=dropout
            ) for i in range(len(num_channels))]
        )

        self.fc = nn.ModuleList([nn.Linear(num_channels[-1], output_size) for _ in quantiles])

    def forward(self, x_num, x_cat):
        x_num = self.num_adapter(x_num)
        transfer_embedded = [emb(x_cat[:, :, i]) for i, emb in enumerate(self.transfer_embedding)]
        mapped_embeddings = [self.mapping_layers[i](transfer_embedded[i]) for i in range(len(self.mapping_layers))]
        x_cat_base_embedded = torch.cat(mapped_embeddings, dim=-1)
        x = torch.cat((x_num, x_cat_base_embedded), dim=-1)
        x = x.permute(0, 2, 1)
        frozen_tcn_output = self.frozen_tcn(x)
        additional_tcn_block = self.additional_tcn(frozen_tcn_output)
        y = additional_tcn_block[:, :, -1]
        y_list = []
        for i in range(len(self.quantiles)):
            y_q = self.fc[i](y)
            y_list.append(y_q)

        y_pred = torch.stack(y_list, dim=-1)
        return y_pred