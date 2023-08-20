import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

class AutoEncoder(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, encode_size, model_name):
        super(AutoEncoder, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size = input_size, num_layers = num_layers, hidden_size = hidden_size, batch_first = True)
        self.model_name = model_name
        self.load_model(model_name)
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
    def train_model(self, train_loader, num_epochs, learning_rate):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            total_loss = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss / len(train_loader):.4f}")
        self.save_model(self.model_name)
    def save_model(self, output_model = "model.pth"):
        torch.save(self.state_dict(), output_model)
    def load_model(self, input_model):
        try:
            self.load_state_dict(torch.load(input_model))
        except FileNotFoundError:
            print("Model not loaded")
        