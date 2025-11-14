import torch.nn as nn


class RNNModel(nn.Module):
    """RNN model for multi-step forecasting"""
    
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.2):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out