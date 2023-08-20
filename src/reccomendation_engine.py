import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

class Reccomend(nn.Module):
    def __init__(self, user_size, product_size):
        self.user_size = user_size
        self.product_size = product_size

        self.fc = nn.Sequential(
            nn.Linear(user_size + product_size, 1000),
            nn.ReLU(),
            nn.Linear(1000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 250),
            nn.ReLU(),
            nn.Linear(250, 1)
        )
    def forward(self, x):
        return self.fc(x)