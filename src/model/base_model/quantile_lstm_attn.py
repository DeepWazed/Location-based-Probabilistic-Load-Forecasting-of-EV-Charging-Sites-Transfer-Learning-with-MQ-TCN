import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, lstm_hidden_size):
        super(Attention, self).__init__()
        self.attention_layer = nn.Linear(lstm_hidden_size, 1)

    def forward(self, lstm_output):
        # Compute scores
        attention_scores = self.attention_layer(lstm_output)  # Shape: [batch_size, sequence_length, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # Shape: [batch_size, sequence_length, 1]
        weighted_sum = torch.sum(lstm_output * attention_weights, dim=1)  # Shape: [batch_size, lstm_hidden_size]
        return weighted_sum, attention_weights

class QuantileLSTMWithAttention(nn.Module):
    def __init__(self, input_size, embedding_dims, lstm_hidden_size, num_layers, output_size, quantiles, dropout=0.2):
        super(QuantileLSTMWithAttention, self).__init__()

        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        self.embeddings = nn.ModuleList([nn.Embedding(num_categories, emb_dim) for num_categories, emb_dim in embedding_dims])
        total_emb_dim = sum([emb_dim for _, emb_dim in embedding_dims])
        self.lstm_input_size = input_size + total_emb_dim
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=lstm_hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attention = Attention(lstm_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc_layers = nn.ModuleList([nn.Linear(lstm_hidden_size, output_size) for _ in range(self.num_quantiles)])

    def forward(self, x_numerical, x_categorical):
        embeddings = [embedding(x_categorical[:, :, i]) for i, embedding in enumerate(self.embeddings)]
        x_cat_embedded = torch.cat(embeddings, dim=-1)
        x_combined = torch.cat([x_numerical, x_cat_embedded], dim=-1)
        
        lstm_out, _ = self.lstm(x_combined)
        context_vector, attention_weights = self.attention(lstm_out)
        context_vector = self.dropout(context_vector)  # dropout after the attention layer Source: Bing

        quantile_preds = [fc(context_vector) for fc in self.fc_layers]
        quantile_preds = torch.stack(quantile_preds, dim=-1)  # Shape: [batch_size, output_size, num_quantiles]

        return quantile_preds