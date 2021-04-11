import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import math
from torch import nn
import torch
import random
from scipy.interpolate import interp1d

# https://www.youtube.com/watch?v=AvKSPZ7oyVg

# GOAL OF THIS CODE
# TRAIN A RNN TO PREDICT THE FUTURE TIME SERIES DATA OF A PENDULUM
# GIVEN A SERIES OF 100 PENDULUM PROBLEMS WITH RANDOM INITIAL CONDITIONS, PREDICT HOW THE PENDULUM WITH BEHAVE IN THE FUTURE

# seed the random functions
random.seed(123)

# data size set that define amount of data sets we will generate to train the network
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

b = 0.3
m = 1

def pendulumODE(t, theta):
    dtheta1 = theta[1]
    dtheta2 = -g/L*math.sin(theta[0])
    return dtheta1, dtheta2

def pendulumODEFriction(t, theta):
    dtheta1 = theta[1]
    dtheta2 = -b/m*theta[1]-g/L*math.sin(theta[0])
    return dtheta1, dtheta2

# sim time
t0, tf = 0, 10
t = np.linspace(t0, tf, 100)

# initilize the arrays used to store the info from the numerical solution
theta = [0 for j in range(DATA_SET_SIZE)]
numericResult = [0 for i in range(DATA_SET_SIZE)]
input_seq = [0 for j in range(DATA_SET_SIZE)]
output_seq = [0 for j in range(DATA_SET_SIZE)]


# generate random data set of input thetas and output thetas and theta dots over a time series 
for i in range(DATA_SET_SIZE):
    theta = [(math.pi/180) * random.randint(-90,90), (math.pi/180) * random.randint(-5,5)]
    numericResult[i] = integrate.solve_ivp(pendulumODEFriction, (t0, tf), theta, "LSODA")
    # print(numericResult[i].y)
    input_seq[i] = numericResult[i].t
    output_seq[i] = numericResult[i].y[:][0]

plt.plot(numericResult[0].t, numericResult[0].y[0])
# plt.show()

# now we should take only a certain amount of data as to reduce times and reduce overfitting data
# first we initilize the 2d arrays to store the information
InputSeqNP = [[0 for j in range(300)] for i in range(DATA_SET_SIZE)]
OutputSeqNP = [[0 for j in range(300)] for i in range(DATA_SET_SIZE)]


# convert the regular arrays to numpy arrays
for i in range(DATA_SET_SIZE):
    InputSeqNP[i] = np.asfarray(input_seq[i])
for i in range(DATA_SET_SIZE):
    OutputSeqNP[i] = np.asfarray(output_seq[i])


# now we downsample the array so the NN gets the same amount of info at all time steps
def downsample(array, npts):
    interpolated = interp1d(np.arange(len(array)), array,
                            axis=0, fill_value='extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled

downsampledInputSeq = [[0 for j in range(100)] for i in range(DATA_SET_SIZE)]
downsampledOutputSeq = [[0 for j in range(100)] for i in range(DATA_SET_SIZE)]

for i in range(DATA_SET_SIZE):
    downsampledInputSeq[i] = downsample(InputSeqNP[i], 200)
for i in range(DATA_SET_SIZE):
    downsampledOutputSeq[i] = downsample(OutputSeqNP[i], 200)

# convert the training data to tensors
trainingDataInput = torch.Tensor(downsampledInputSeq)
trainingDataOutput = torch.Tensor(downsampledOutputSeq)


# generate the test function that will be used to test the NN
thetaTest = [(math.pi/180) * random.randint(-90,90), (math.pi/180) * random.randint(-5,5)]
testNumericalResult = integrate.solve_ivp(pendulumODEFriction, (t0, tf/2), thetaTest, "LSODA")
testInnputSeq = testNumericalResult.t
testOutputSeq = testNumericalResult.y[0]

testInputSeqNP = np.asfarray(testInnputSeq)
testOutputSeqNP = np.asfarray(testOutputSeq)

downsampledTestInputSeq = downsample(testInputSeqNP, 200)
downsampledTestOutputSeq = downsample(testOutputSeqNP, 200)

testingDataInput = torch.Tensor(downsampledTestInputSeq)
testingDataOutput = torch.Tensor(downsampledTestOutputSeq)


# ------------------------------------------------------------------------
## RNN

# hyperparameters
# from stanford poster example (https://web.stanford.edu/class/archive/cs/cs221/cs221.1196/posters/18560035.pdf)
n_epochs = 30
n_epochs = 2
# lr = 5*(10**-5)
lr = 0.08
input_size = 2
output_size = 2
num_layers = 2
hidden_size = 10

# defining the model class
class pendulumRNN(nn.Module):
    def __init__(self, hidden_dim):
        super(pendulumRNN, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm1 = nn.LSTMCell(1,self.hidden_dim)
        self.lstm2 = nn.LSTMCell(self.hidden_dim,self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim,1)

    def forward(self,x,future=0):
        outputs=[]
        n_samples = x.size(0)
        h_t = torch.zeros(n_samples,self.hidden_dim, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden_dim, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.hidden_dim, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.hidden_dim, dtype=torch.float32)

        for input_t in x.split(1, dim = 1):
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

# initilizing the model, criterion, and optimizer for the data
model = pendulumRNN(hidden_size)
criterion = nn.MSELoss()
optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)

# defining the back prop function
def closure():
    optimizer.zero_grad()
    out = model(trainingDataInput)
    loss = criterion(out, trainingDataOutput)
    print("loss", loss.item())
    loss.backward()
    return loss

# training loop
for epoch in range(n_epochs):
    print("Step",epoch)
    optimizer.step(closure)

