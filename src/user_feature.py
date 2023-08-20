import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

class AutoEncoder(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, encode_size):
        super(AutoEncoder, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # self.loss = nn.MSELoss()
        # self.optimizer = optim.Adam(self.parameters(), lr = 0.001)
        self.lstm = nn.LSTM(input_size = input_size, num_layers = num_layers, hidden_size = hidden_size, batch_first = True)

        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, 1000),
            nn.ReLU(),
            nn.Linear(1000, encode_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encode_size, 1000),
            nn.ReLU(),
            nn.Linear(1000, input_size)
        )
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
    def encode(self, x):
        _, (out, _) = self.lstm(x)
        out = out[-1]
        out = self.encoder(out)
        return out

