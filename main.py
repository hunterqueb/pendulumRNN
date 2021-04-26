import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import integrate
import math
from torch import nn
import torch
import random
from scipy.interpolate import interp1d
import os.path

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
L = 0.5
g = 9.81

b = 0.1
m = 1

def pendulumODE(theta, t):
    dtheta1 = theta[1]
    dtheta2 = -g/L*math.sin(theta[0])
    return [dtheta1, dtheta2]

def pendulumODEFriction(theta, t):
    dtheta1 = theta[1]
    dtheta2 = -b/m*theta[1]-g/L*math.sin(theta[0])
    return [dtheta1, dtheta2]

# sim time
t0, tf = 0, 10

SAMPLE_SIZE = tf/TIME_STEP

t = np.arange(t0, tf, TIME_STEP)

# initilize the arrays used to store the info from the numerical solution
theta = [0 for i in range(DATA_SET_SIZE)]
numericResult = [0 for i in range(DATA_SET_SIZE)]
output_seq = [0 for i in range(DATA_SET_SIZE)]
# generate random data set of input thetas and output thetas and theta dots over a time series 
for i in range(DATA_SET_SIZE):
    theta = [(math.pi/180) * random.randint(70,90), (math.pi/180) * 0]
    # numericResult[i] = integrate.solve_ivp(pendulumODEFriction, (t0, tf), theta, "LSODA")
    numericResult = integrate.odeint(pendulumODEFriction, theta, t)
    output_seq[i] = numericResult[:,0]
    if i == DATA_SET_SIZE-1:
        actualResultFull = integrate.odeint(pendulumODEFriction, theta, np.arange(t0, 2*tf, TIME_STEP))
        actualResult = actualResultFull[:, 0]


# convert the python list to numpy array
output_seq = np.asfarray(output_seq)


# now convert the training data to tensors
trainingDataInput = torch.from_numpy(output_seq[NUM_TEST_SIZE:, :-1])
trainingDataOutput = torch.from_numpy(output_seq[NUM_TEST_SIZE:, 1:])

testingDataInput = torch.from_numpy(output_seq[:NUM_TEST_SIZE, :-1])
testingDataOutput = torch.from_numpy(output_seq[:NUM_TEST_SIZE, 1:])

trainingDataInput = trainingDataInput.float()
trainingDataOutput = trainingDataOutput.float()

testingDataInput = testingDataInput.float()
testingDataOutput = testingDataOutput.float()

# SIMPLE SINISOID TESTING
# T = 20
# L = 1000
# N = 100

# x = np.empty((N, L), 'int64')
# x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
# data = np.sin(x / 1.0 / T).astype('float64')

# trainingDataInput = torch.from_numpy(data[3:, :-1])
# trainingDataOutput = torch.from_numpy(data[3:, 1:])

# testingDataInput = torch.from_numpy(data[:3, :-1])
# testingDataOutput = torch.from_numpy(data[:3, 1:])

# trainingDataInput = trainingDataInput.float()
# trainingDataOutput = trainingDataOutput.float()

# testingDataInput = testingDataInput.float()
# testingDataOutput = testingDataOutput.float()


def drawPrediction(yi, color):
    plt.plot(np.arange(trainingDataInput.size(1)), yi[:trainingDataInput.size(1)], color, linewidth=2.0)
    plt.plot(np.arange(trainingDataInput.size(1), trainingDataInput.size(1) + future), yi[trainingDataInput.size(1):], color + ':', linewidth=2.0)

def drawPlot(yi, color):
    plt.plot(yi, color, linewidth=2.0)


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
n_epochs = 30
n_epochs = 100
lr = 5*(10**-5)
lr = 0.004
lr = 0.08
lr = 0.1
input_size = 1
output_size = 1
num_layers = 3
hidden_size = 50
momentum = 0.9

# defining the model class
class pendulumRNN(nn.Module):
    
    def __init__(self, hidden_dim):
        super(pendulumRNN, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm1 = nn.LSTMCell(1,self.hidden_dim)
        self.lstm2 = nn.LSTMCell(self.hidden_dim,self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim,1)

    def forward(self,input,future=0):
        outputs=[]
        n_samples = input.size(0)
        h_t = torch.zeros(n_samples,self.hidden_dim, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden_dim, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.hidden_dim, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.hidden_dim, dtype=torch.float32)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t,(h_t,c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        outputs = torch.cat(outputs, dim = 1)
        return outputs


class pendulumRNN3(nn.Module):
    def __init__(self, hidden_dim):
        super(pendulumRNN3, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm1 = nn.LSTMCell(1, self.hidden_dim)
        self.lstm2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.lstm3 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, 1)

    def forward(self, input, future=0):
        outputs = []
        n_samples = input.size(0)
        h_t = torch.zeros(n_samples, self.hidden_dim, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden_dim, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.hidden_dim, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.hidden_dim, dtype=torch.float32)
        h_t3 = torch.zeros(n_samples, self.hidden_dim, dtype=torch.float32)
        c_t3 = torch.zeros(n_samples, self.hidden_dim, dtype=torch.float32)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            output = self.linear(h_t3)
            outputs.append(output)

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            output = self.linear(h_t3)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs

# initilizing the model, criterion, and optimizer for the data
model = pendulumRNN(hidden_size)
criterion = nn.MSELoss()
optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)


# training loop
for epoch in range(n_epochs):
    print("Step",epoch)
    file.write("Step " +  str(epoch) + "\n")
    def closure():
        # defining the back prop function
        optimizer.zero_grad()
        out = model(trainingDataInput)
        loss = criterion(out, trainingDataOutput)
        print("loss", loss.item())
        file.write("loss: " + str(loss.item()) + "\n")
        loss.backward()
        return loss
    optimizer.step(closure)

    with torch.no_grad():
        future = int(SAMPLE_SIZE)
        pred = model(testingDataInput, future=future)
        loss = criterion(pred[:, :-future], testingDataOutput)
        print("test loss", loss.item())
        file.write("test loss: " + str(loss.item()) + "\n")
        pendulumPrediction = pred.detach().numpy()
        # this is our prediction array
    
    # drawPrediction the result
    plt.figure(figsize=(30, 10))
    plt.title(
        'Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    plt.xlabel('time', fontsize=20)
    plt.ylabel('theta', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    drawPrediction(pendulumPrediction[0], 'r')
    drawPlot(actualResult,'b')
    # drawPrediction(pendulumPrediction[1], 'g')
    # drawPrediction(pendulumPrediction[2], 'b')
    plt.savefig('predict%d.pdf' % epoch)
    plt.close()

file.close()
torch.save(model.state_dict(), open('trainedModel'+str(reportNum)+'.pt', 'wb'))
