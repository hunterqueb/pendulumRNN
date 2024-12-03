# new example using the knowledge i have gained....

# lets do a duffing!
# adam optimizer
# criterion = F.smooth_l1_loss

# https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
'''
usually, time seties oreduction is done on a window.
given data from time [t - w,t], you predict t + m where m is any timestep into the future.
w governs how much data you can look at to make a predition, called the look back period.


'''
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import random
import torch.nn.functional as F
import torch.utils.data as data

from qutils.integrators import myRK4Py
from qutils.mlExtras import findDecAcc, plotSuperWeight
from qutils.ml import create_datasets, genPlotPrediction,LSTMSelfAttentionNetwork, transferLSTM

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

def linPendulumODE(t,theta,p=None):
    m = 1
    k = 1
    dtheta1 = theta[1]
    dtheta2 = -k/m*(theta[0])
    return np.array([dtheta1, dtheta2])


L = 10
g = 9.81


def pendulumODE(t,theta,p=None):
    dtheta1 = theta[1]
    dtheta2 = -g/L*math.sin(theta[0])
    return np.array([dtheta1, dtheta2])


b = 0.1


def pendulumODEFriction(t,theta,p=None):
    m = 1
    dtheta1 = theta[1]
    dtheta2 = -b/m*theta[1]-g/L*math.sin(theta[0])
    return np.array([dtheta1, dtheta2])


c = 0.05
w = 1.4
k = 0.7

# strange values from my discussion with pugal
c = 0.2
w = 1.3
k = 2


def duffingOscillatorODE(t,y, p=[c, w**2, k**2]):
    dydt1 = y[1]
    dydt2 = -(p[0]*y[1] + p[1]*y[0] + p[2]*y[0]**3)

    return np.array([dydt1, dydt2])


sysfuncptr = linPendulumODE
# sim time
t0, tf = 0, 2*np.pi/w * 5

t = np.arange(t0, tf, TIME_STEP)
degreesOfFreedom = 2
# initilize the arrays used to store the info from the numerical solution
theta = np.zeros((degreesOfFreedom,DATA_SET_SIZE))
output_seq = np.zeros((len(t),degreesOfFreedom))

# generate random data set of input thetas and output thetas and theta dots over a time series
theta = np.array([random.uniform(0.84, 0.86), 0])
# numericResult[i] = integrate.solve_ivp(pendulumODEFriction, (t0, tf), theta, "LSODA")
numericResult = myRK4Py(sysfuncptr,theta,t,paramaters=np.array([c, w**2, k**2]))
output_seq = numericResult

linPendNR = numericResult

# hyperparameters
n_epochs = 50
# lr = 5*(10**-5)
# lr = 0.85
lr = 0.8
lr = 0.08
lr = 0.004
lr = 0.001
input_size = degreesOfFreedom
output_size = degreesOfFreedom
num_layers = 1
hidden_size = 50
p_dropout = 0.0
lookback = 1
p_motion_knowledge = 0.2

'''
with a lookback window of 4, the input to the network is [t,t+1,t+2,t+3] and the output is [t+1,t+2,t+3,t+4]

for long time series, overlapping windows can be created.
A time series of n time steps can produce roughly n windows because a window can start from any time step as long as the window does 
not go beyond the boundary of the time series. Within one window, there are multiple consecutive time steps of values. 
In each time step, there can be multiple features.
'''


train_size = int(len(output_seq) * p_motion_knowledge)
test_size = len(output_seq) - train_size

train_in,train_out,test_in,test_out = create_datasets(output_seq,1,train_size,device)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

# initilizing the model, criterion, and optimizer for the data
model = LSTMSelfAttentionNetwork(input_size,hidden_size,output_size,num_layers, p_dropout).double().to(device)
# model = LSTM(input_size,hidden_size,output_size,num_layers, p_dropout).double().to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = F.smooth_l1_loss
# criterion = torch.nn.MSELoss()
# Define the mean absolute error (MAE) loss function
# mae_loss = F.l1_loss(predicted, target)

# Define the Huber loss function with delta=1.0
# huber_loss = F.smooth_l1_loss(predicted, target, reduction='mean', delta=1.0)

def plotPredition(epoch,err=None):
        train_plot, test_plot = genPlotPrediction(model,output_seq,train_in,test_in,train_size,1)

        fig, (ax1, ax2) = plt.subplots(2,1)
        # plot
        ax1.plot(t,output_seq[:,0], c='b',label = 'True Motion')
        ax1.plot(t,train_plot[:,0], c='r',label = 'Training Region')
        ax1.plot(t,test_plot[:,0], c='g',label = 'Predition')
        ax1.set_title('LSTM Solution to Linear Oscillator')
        # ax1.xlabel('time (sec)')
        ax1.set_ylabel('x (m)')
        # plt.legend(loc="lower left")

        ax2.plot(t,output_seq[:,1], c='b',label = 'True Motion')
        ax2.plot(t,train_plot[:,1], c='r',label = 'Training Region')
        ax2.plot(t,test_plot[:,1], c='g',label = 'Predition')
        ax2.set_xlabel('time (sec)')
        ax2.set_ylabel('xdot (m/s)')
        plt.legend(loc="lower left")
        plt.tight_layout()

        plt.savefig('predict/predict%d.png' % epoch)
        plt.close()

        if err is not None:
            fig, (ax1, ax2) = plt.subplots(2,1)
            ax1.plot(err)
            ax2.plot(np.average(err,axis=0)*np.ones(err.shape))
            plt.show()
        # filter out nan values for better post processing
        train_plot = train_plot[~np.isnan(train_plot)]
        test_plot = test_plot[~np.isnan(test_plot)]

        trajPredition = np.concatenate((train_plot,test_plot))

        return trajPredition.reshape((len(trajPredition),1))

for epoch in range(n_epochs):

    trajPredition = plotPredition(epoch)

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
        y_pred_train = model(train_in)
        train_loss = np.sqrt(criterion(y_pred_train, train_out).cpu())
        y_pred_test = model(test_in)
        test_loss = np.sqrt(criterion(y_pred_test, test_out).cpu())

        decAcc, err1 = findDecAcc(train_out,y_pred_train,printOut=False)
        decAcc, err2 = findDecAcc(test_out,y_pred_test)
        err = np.concatenate((err1,err2),axis=0)

    print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))

plotPredition(epoch+1,err = err)

"""
TRANSFER LEARN TO NEW, NONLINEAR SYSTEM ON DIFFERENT INITIAL CONDITIONS AND DIFFERENT TIME PERIOD AND DIFFERENT TIME STEP
"""

TIME_STEP = 0.005

# transfer to different system
newModel = LSTMSelfAttentionNetwork(input_size,hidden_size,output_size,num_layers, p_dropout).double().to(device)
trainableLayer = [True, True, False]
newModel = transferLSTM(model,newModel)

sysfuncptr = duffingOscillatorODE
# sim time
t0, tf = 0, 2*np.pi/w * 10

t = np.arange(t0, tf, TIME_STEP)
degreesOfFreedom = 2

# initilize the arrays used to store the info from the numerical solution
theta = np.zeros((degreesOfFreedom,DATA_SET_SIZE))
output_seq = np.zeros((len(t),degreesOfFreedom))

# generate random data set of input thetas and output thetas and theta dots over a time series
theta = np.array([random.uniform(-0.4, -0.5), random.uniform(0.1, 0.5)])

numericResult = myRK4Py(sysfuncptr,theta,t,paramaters=np.array([c, w**2, k**2]))
output_seq = numericResult

duffPendNR = numericResult

train_size = int(len(output_seq) * p_motion_knowledge)
test_size = len(output_seq) - train_size

train_in,train_out,test_in,test_out = create_datasets(output_seq,1,train_size,device)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

n_epochs = 10
lr = 0.01
input_size = degreesOfFreedom
output_size = degreesOfFreedom
num_layers = 1
hidden_size = 50
p_dropout = 0.0
lookback = 1
p_motion_knowledge = 0.2

optimizer = torch.optim.Adam(newModel.parameters(),lr=lr)

def plotNewPredition(epoch,err=None):
        with torch.no_grad():
            train_plot = np.ones_like(output_seq) * np.nan
            y_pred = model(train_in)
            y_pred = y_pred[:, -1, :]
            train_plot[1:train_size+1] = model(train_in)[:, -1, :].cpu()
            # shift test predictions for plotting
            test_plot = np.ones_like(output_seq) * np.nan
            test_plot[train_size+1:] = model(test_in)[:, -1, :].cpu()

        fig, (ax1, ax2) = plt.subplots(2,1)
        # plot
        ax1.plot(t,output_seq[:,0], c='b',label = 'True Motion')
        ax1.plot(t,train_plot[:,0], c='r',label = 'Training Region')
        ax1.plot(t,test_plot[:,0], c='g',label = 'Predition')
        ax1.set_title('LSTM Solution to Duffing Oscillator')
        # ax1.xlabel('time (sec)')
        ax1.set_ylabel('x (m)')
        # plt.legend(loc="lower left")

        ax2.plot(t,output_seq[:,1], c='b',label = 'True Motion')
        ax2.plot(t,train_plot[:,1], c='r',label = 'Training Region')
        ax2.plot(t,test_plot[:,1], c='g',label = 'Predition')
        ax2.set_xlabel('time (sec)')
        ax2.set_ylabel('xdot (m/s)')
        plt.legend(loc="lower left")
        plt.tight_layout()

        plt.savefig('predict/newPredict%d.png' % epoch)
        plt.close()

        if err is not None:
            fig, (ax1, ax2) = plt.subplots(2,1)
            ax1.plot(err)
            ax2.plot(np.average(err,axis=0)*np.ones(err.shape))
            plt.show()
        # filter out nan values for better post processing
        train_plot = train_plot[~np.isnan(train_plot)]
        test_plot = test_plot[~np.isnan(test_plot)]

        trajPredition = np.concatenate((train_plot,test_plot))

        return trajPredition.reshape((len(trajPredition),1))


for epoch in range(n_epochs):

    trajPredition = plotNewPredition(epoch)

    newModel.train()
    for X_batch, y_batch in loader:
        y_pred = newModel(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    newModel.eval()
    with torch.no_grad():
        y_pred_train = newModel(train_in)
        train_loss = np.sqrt(criterion(y_pred_train, train_out).cpu())
        y_pred_test = newModel(test_in)
        test_loss = np.sqrt(criterion(y_pred_test, test_out).cpu())

        decAcc, err1 = findDecAcc(train_out,y_pred_train,printOut=False)
        decAcc, err2 = findDecAcc(test_out,y_pred_test)
        err = np.concatenate((err1,err2),axis=0)

    print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))

plotNewPredition(epoch+1,err)

plt.figure()
plotSuperWeight(model,newPlot=False)
plotSuperWeight(newModel,newPlot=False)
plt.grid()
plt.tight_layout()
plt.show()