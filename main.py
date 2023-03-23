import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import math
from torch import nn
import torch
import random
from scipy.interpolate import interp1d
import os.path
import torch.nn.functional as F


# https://www.youtube.com/watch?v=AvKSPZ7oyVg

# GOAL OF THIS CODE
# TRAIN A RNN TO PREDICT THE FUTURE TIME SERIES DATA OF A PENDULUM
# GIVEN A SERIES OF 100 PENDULUM PROBLEMS WITH RANDOM INITIAL CONDITIONS, PREDICT HOW THE PENDULUM WITH BEHAVE IN THE FUTURE

# seed the random functions
random.seed(123)

# data size set that define amount of data sets we will generate to train the network
DATA_SET_SIZE = 100
NUM_TEST_SIZE = 1
DATA_SET_SIZE = DATA_SET_SIZE + NUM_TEST_SIZE
TIME_STEP = 0.01

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# ------------------------------------------------------------------------
## NUMERICAL SOLUTION


m = 1
k = 1


def linPendulumODE(theta, t):
    dtheta1 = theta[1]
    dtheta2 = -k/m*(theta[0])
    return [dtheta1, dtheta2]


L = 10
g = 9.81


def pendulumODE(theta, t):
    dtheta1 = theta[1]
    dtheta2 = -g/L*math.sin(theta[0])
    return [dtheta1, dtheta2]


b = 0.1


def pendulumODEFriction(theta, t):
    dtheta1 = theta[1]
    dtheta2 = -b/m*theta[1]-g/L*math.sin(theta[0])
    return [dtheta1, dtheta2]


c = 0
w = 1.4
k = 0.7


def duffingOscillatorODE(y, t, p=[c, w**2, k**2]):
    dydt1 = y[1]
    dydt2 = -(p[0]*y[1] + p[1]*y[0] + p[2]*y[0]**3)

    return [dydt1, dydt2]


sysfuncptr = duffingOscillatorODE
# sim time
t0, tf = 0, 2*np.pi/w

SAMPLE_SIZE = tf/TIME_STEP

t = np.arange(t0, tf, TIME_STEP)

# initilize the arrays used to store the info from the numerical solution
theta = [0 for i in range(DATA_SET_SIZE)]
numericResult = [0 for i in range(DATA_SET_SIZE)]
output_seq = [0 for i in range(DATA_SET_SIZE)]
# generate random data set of input thetas and output thetas and theta dots over a time series
for i in range(DATA_SET_SIZE):
    theta = [random.uniform(0.84, 0.86), (math.pi/180) * 0]
    # numericResult[i] = integrate.solve_ivp(pendulumODEFriction, (t0, tf), theta, "LSODA")
    numericResult = integrate.odeint(sysfuncptr, theta, t)
    output_seq[i] = numericResult[:, 0]
    if i == DATA_SET_SIZE-1:
        actualResultFull = integrate.odeint(
            sysfuncptr, theta, np.arange(t0, 2*tf, TIME_STEP))
        actualResult = actualResultFull[:, 0]


# convert the python list to numpy array
output_seq = np.asfarray(output_seq)


# now convert the training data to tensors
# output_seq = 101,1000
trainingDataInput = torch.from_numpy(
    output_seq[NUM_TEST_SIZE:, :-1])  # 100, 999
trainingDataOutput = torch.from_numpy(
    output_seq[NUM_TEST_SIZE:, 1:])  # 100, 999

testingDataInput = torch.from_numpy(output_seq[:NUM_TEST_SIZE, :-1])  # 1, 999
testingDataOutput = torch.from_numpy(output_seq[:NUM_TEST_SIZE, 1:])  # 1, 999

trainingDataInput = trainingDataInput.float().to(device)
trainingDataOutput = trainingDataOutput.float().to(device)
testingDataInput = testingDataInput.float().to(device)
testingDataOutput = testingDataOutput.float().to(device)


def drawPrediction(yi, color):
    plt.plot(np.arange(t0, tf-TIME_STEP, TIME_STEP),
             yi[:trainingDataInput.size(1)], color, linewidth=2.0, label='Direct Training Output')
    plt.plot(np.arange(tf, tf*2-TIME_STEP, TIME_STEP),
             yi[trainingDataInput.size(1):], color + ':', linewidth=2.0, label='Predicted Motion')


def drawPlot(yi, color):
    plt.plot(np.arange(t0, 2*tf, TIME_STEP), yi,
             color, linewidth=2.0, label='True Motion')


# ------------------------------------------------------------------------
## FILE IO
reportNum = 0
reportCreated = False
while(not reportCreated):
    if os.path.isfile("report" + str(reportNum) + ".txt"):
        reportNum += 1
    else:
        file = open("report" + str(reportNum) + ".txt", "w")
        reportCreated = True


# ------------------------------------------------------------------------
## RNN

# hyperparameters
# from stanford poster example (https://web.stanford.edu/class/archive/cs/cs221/cs221.1196/posters/18560035.pdf)
n_epochs = 70
# n_epochs = 100
lr = 5*(10**-5)
lr = 0.004
# lr = 0.08
# lr = 0.85
lr = 0.7
input_size = 1
output_size = 1
# num_layers = 3
hidden_size = 51
p_dropout = 0

# defining the model class


class pendulumRNN(nn.Module):

    def __init__(self, hidden_dim, dropout_prob=0):
        super(pendulumRNN, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm1 = nn.LSTMCell(1, self.hidden_dim)
        self.lstm2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, input, future=0):
        outputs = []
        n_samples = input.size(0)
        h_t = torch.zeros(n_samples, self.hidden_dim,
                          dtype=torch.float32, device=device)
        c_t = torch.zeros(n_samples, self.hidden_dim,
                          dtype=torch.float32, device=device)
        h_t2 = torch.zeros(n_samples, self.hidden_dim,
                           dtype=torch.float32, device=device)
        c_t2 = torch.zeros(n_samples, self.hidden_dim,
                           dtype=torch.float32, device=device)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t2 = self.dropout(h_t2)
            output = self.linear(h_t2)
            outputs.append(output)

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t2 = self.dropout(h_t2)
            output = self.linear(h_t2)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs


class pendulumRNN3(nn.Module):
    def __init__(self, hidden_dim, dropout_prob=0.0):
        super(pendulumRNN3, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm1 = nn.LSTMCell(1, self.hidden_dim)
        self.lstm2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.lstm3 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, input, future=0):
        outputs = []
        n_samples = input.size(0)
        h_t = torch.zeros(n_samples, self.hidden_dim,
                          dtype=torch.float32, device=device)
        c_t = torch.zeros(n_samples, self.hidden_dim,
                          dtype=torch.float32, device=device)
        h_t2 = torch.zeros(n_samples, self.hidden_dim,
                           dtype=torch.float32, device=device)
        c_t2 = torch.zeros(n_samples, self.hidden_dim,
                           dtype=torch.float32, device=device)
        h_t3 = torch.zeros(n_samples, self.hidden_dim,
                           dtype=torch.float32, device=device)
        c_t3 = torch.zeros(n_samples, self.hidden_dim,
                           dtype=torch.float32, device=device)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            h_t3 = self.dropout(h_t3)
            output = self.linear(h_t3)
            outputs.append(output)

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            h_t3 = self.dropout(h_t3)
            output = self.linear(h_t3)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs


# initilizing the model, criterion, and optimizer for the data
model = pendulumRNN3(hidden_size, p_dropout).to(device)
criterionMSE = nn.MSELoss()
optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, tolerance_grad=1e-8, tolerance_change=1e-10)
criterion = F.smooth_l1_loss
# Define the mean absolute error (MAE) loss function
# mae_loss = F.l1_loss(predicted, target)

# Define the Huber loss function with delta=1.0
# huber_loss = F.smooth_l1_loss(predicted, target, reduction='mean', delta=1.0)


# training loop
for epoch in range(n_epochs):
    print("Step", epoch)
    file.write("Step " + str(epoch) + "\n")

    def closure():
        # defining the back prop function
        optimizer.zero_grad()
        out = model(trainingDataInput)
        loss = criterion(out, trainingDataOutput,reduction='mean', beta=1e-5, size_average=True)
        print("     loss", loss.item())
        file.write("     loss: " + str(loss.item()) + "\n")
        loss.backward()
        return loss
    optimizer.step(closure)

    with torch.no_grad():
        future = int(SAMPLE_SIZE)
        pred = model(testingDataInput, future=future)
        loss = criterionMSE(pred[:, :-future], testingDataOutput)
        print("MSE  loss", loss.item())
        file.write("test loss: " + str(loss.item()) + "\n")
        pendulumPrediction = pred.cpu().detach().numpy()
        # this is our prediction array

    # drawPrediction the result
    if epoch % 5 == 0 or epoch == n_epochs-1:
        plt.figure(figsize=(30, 10))
        plt.title('LSTM Solution to Duffing Oscillator', fontsize=30)
        plt.xlabel('time (sec)', fontsize=20)
        plt.ylabel('x (m)', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        drawPrediction(pendulumPrediction[0], 'r')
        drawPlot(actualResult, 'k')
        plt.legend(fontsize=15)
        # drawPrediction(pendulumPrediction[1], 'g')
        # drawPrediction(pendulumPrediction[2], 'b')
        plt.savefig('predict%d.pdf' % epoch)
        plt.close()

file.close()
torch.save(model.state_dict(), open('trainedModel'+str(reportNum)+'.pt', 'wb'))
