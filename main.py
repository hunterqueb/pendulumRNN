import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import math
from torch import nn
import torch
import random

random.seed(123)

DATA_SET_SIZE = 100

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


## NUMERICAL SOLUTION
L = 0.5
g = 9.81

b = 1
m = 1

def pendulumODE(t, theta):
    dtheta1 = theta[1]
    dtheta2 = -g/L*math.sin(theta[0])
    return dtheta1, dtheta2

def pendulumODEFriction(t, theta):
    dtheta1 = theta[1]
    dtheta2 = -b/m*theta[1]-g/L*math.sin(theta[0])
    return dtheta1, dtheta2

t0, tf = 0, 10
t = np.linspace(t0, tf, 100)


theta = [[0 for i in range(2)] for j in range(DATA_SET_SIZE)]
input_seq = [[0 for i in range(2)] for j in range(DATA_SET_SIZE)]
output_seq = [[0 for i in range(2)] for j in range(DATA_SET_SIZE)]
numericResult = [0 for i in range(DATA_SET_SIZE)]


# generate randome data set of input thetas and output thetas and theta dots over a time series 
for i in range(DATA_SET_SIZE):
    theta = [(math.pi/180) * random.randint(-90,90), (math.pi/180) * random.randint(-5,5)]
    numericResult[i] = integrate.solve_ivp(pendulumODEFriction, (t0, tf), theta, "LSODA")
    for j in range(2):
        input_seq[i][j] = theta[j]
    # print(numericResult[i].y)
    output_seq[i][:] = numericResult[i].y

for i in range(DATA_SET_SIZE):
    input_seq[i][:] = torch.Tensor(input_seq[i][:])
    output_seq[i][:] = torch.Tensor(output_seq[i][:])


# hyperparameters
n_epochs = 2
lr = 0.01
input_size = 2
output_size = 2
sequence_length = max(input_seq)
num_layers = 2
hidden_size = 200



class pendulumRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(pendulumRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim*sequence_length, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        hidden = self.init_hidden(batch_size)

        out, hidden = self.rnn(x, hidden)

        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):

        hidden = torch.zeros(self.n_layers, batch_size,
                             self.hidden_dim).to(device)

        return hidden


model = pendulumRNN(input_size=input_size, output_size=output_size,
              hidden_dim=hidden_size, n_layers=num_layers)
# We'll also set the model to the device that we defined earlier (default is CPU)
model = model.to(device)

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    for batch in range(input_seq[:][1]):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backwards
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()



