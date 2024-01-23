import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # Pack the sequence
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Forward pass through LSTM
        packed_output, _ = self.lstm(packed_input)

        # Unpack the sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # We take the output of the last time step for each sequence
        idx = (torch.LongTensor(lengths) - 1).view(-1, 1).expand(len(lengths), output.size(2))
        idx = idx.unsqueeze(1).to(x.device)
        last_output = output.gather(1, idx).squeeze(1)

        # Pass through the output layer
        out = self.fc(last_output)
        return out

# Model parameters
input_size = 10  # Number of features
hidden_size = 50
num_layers = 2
output_size = 1  # For example, binary classification

# Create model instance
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Dummy input data (batch_size, seq_length, input_size)
x_dummy = torch.randn(5, 7, input_size)

# Lengths of each sequence in the batch
lengths_dummy = torch.LongTensor([7, 6, 5, 4, 3])  # Example lengths

# Forward pass (model outputs predictions for each sequence)
outputs = model(x_dummy, lengths_dummy)
print(outputs)
