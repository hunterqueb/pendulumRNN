# new example using the knowledge i have gained....

# lets do a duffing!
# adam optimizer
# criterion = F.smooth_l1_loss


import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import random
import torch.nn.functional as F
from nets import LSTMSelfAttentionNetwork, create_dataset, LSTM
from quebutils.integrators import myRK4Py
import torch.utils.data as data

# seed any random functions
random.seed(123)

# data size set that define amount of data sets we will generate to train the network
DATA_SET_SIZE = 1
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


def duffingOscillatorODE(t,y, p=[c, w**2, k**2]):
    dydt1 = y[1]
    dydt2 = -(p[0]*y[1] + p[1]*y[0] + p[2]*y[0]**3)

    return np.array([dydt1, dydt2])


sysfuncptr = duffingOscillatorODE
# sim time
t0, tf = 0, 2*np.pi/w * 4

t = np.arange(t0, tf, TIME_STEP)
degreesOfFreedom = 2
# initilize the arrays used to store the info from the numerical solution
theta = np.zeros((degreesOfFreedom,DATA_SET_SIZE))
output_seq = np.zeros((len(t),DATA_SET_SIZE))

# generate random data set of input thetas and output thetas and theta dots over a time series
for i in range(DATA_SET_SIZE):
    theta[:,i] = [random.uniform(0.84, 0.86), 0]
    # numericResult[i] = integrate.solve_ivp(pendulumODEFriction, (t0, tf), theta, "LSODA")
    numericResult = myRK4Py(sysfuncptr,theta[:,i],t,paramaters=np.array([c, w**2, k**2]))
    output_seq[:,i] = numericResult[:, 0]


train_size = int(len(output_seq) * 0.67)
test_size = len(output_seq) - train_size

train, test = output_seq[:train_size], output_seq[train_size:]

lookback = 2

train_in,train_out = create_dataset(train,device,lookback=lookback)
test_in,test_out = create_dataset(test,device,lookback=lookback)


# hyperparameters
n_epochs = 50
# lr = 5*(10**-5)
# lr = 0.85
lr = 0.004
lr = 0.8
lr = 0.08
input_size = 1
output_size = 1
num_layers = 1
hidden_size = 50
p_dropout = 0

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

# initilizing the model, criterion, and optimizer for the data
model = LSTMSelfAttentionNetwork(input_size,hidden_size,output_size,num_layers, p_dropout).double().to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = F.smooth_l1_loss
# criterion = nn.MSELoss()
# Define the mean absolute error (MAE) loss function
# mae_loss = F.l1_loss(predicted, target)

# Define the Huber loss function with delta=1.0
# huber_loss = F.smooth_l1_loss(predicted, target, reduction='mean', delta=1.0)

def plotPredition(epoch):
        with torch.no_grad():
            # shift train predictions for plotting
            train_plot = np.ones_like(output_seq) * np.nan
            y_pred = model(train_in)
            y_pred = y_pred[:, -1, :]
            train_plot[lookback:train_size] = model(train_in)[:, -1, :]
            # shift test predictions for plotting
            test_plot = np.ones_like(output_seq) * np.nan
            test_plot[train_size+lookback:len(output_seq)] = model(test_in)[:, -1, :]
        # plot
        plt.plot(t,output_seq, c='b',label = 'True Motion')
        plt.plot(t,train_plot, c='r',label = 'Training Data')
        plt.plot(t,test_plot, c='g',label = 'Predition')

        plt.title('LSTM Solution to Duffing Oscillator')
        plt.xlabel('time (sec)')
        plt.ylabel('x (m)')
        plt.legend(loc="lower left")
        
        plt.savefig('predict/predict%d.png' % epoch)
        plt.close()


for epoch in range(n_epochs):

    plotPredition(epoch)

    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    with torch.no_grad():
        y_pred = model(train_in)
        train_loss = np.sqrt(criterion(y_pred, train_out))
        y_pred = model(test_in)
        test_loss = np.sqrt(criterion(y_pred, test_out))

    print("Epoch %d: train loss %.4f, test loss %.4f" % (epoch, train_loss, test_loss))

plotPredition(epoch+1)
