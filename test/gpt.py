import torch
import torch.nn as nn
import torch.optim as optim

class MotionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MotionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, future=0):
        # Initialize hidden and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass the input sequence through the LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Initialize the output sequence
        pred = torch.zeros(x.size(0), future, x.size(2)).to(x.device)

        # Use the last hidden state as the initial input for prediction
        input = out[:, -1, :].unsqueeze(1)

        # Generate future predictions recursively
        for i in range(future):
            out, (h0, c0) = self.lstm(input, (h0, c0))
            pred[:, i:i+1, :] = self.linear(out)
            input = out

        return pred

# Generate synthetic motion data (x,y,z coordinates of an object over time)
data_length = 100
future = 10
input_size = 3
hidden_size = 64
output_size = 3
num_layers = 2
batch_size = 1

data = torch.randn(batch_size, data_length, input_size)

# Split the data into input and target sequences
input_seq = data[:, :data_length-future, :]
target_seq = data[:, future:, :]

# Create the RNN model and define the loss function and optimizer
model = MotionRNN(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model on the input sequence
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    pred = model(input_seq, future=future)
    loss = criterion(pred, target_seq)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Generate future predictions using the trained model
with torch.no_grad():
    future_seq = model(input_seq, future=future)

# Plot the input and target sequences, as well as the predicted future sequence
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(data[0, :, 0], label='Input Sequence')
plt.plot(range(data_length, data_length+future), future_seq[0, :, 0], label='Predicted Sequence')
plt.plot(range(data_length, data_length+future), target_seq[0, :, 0], label='Target Sequence')
plt.legend()
plt.show()