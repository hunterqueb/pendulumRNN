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

from quebUtils.integrators import myRK4Py
from quebUtils.mlExtras import findDecAcc

from nets import LSTMSelfAttentionNetwork, create_dataset, LSTM, transferLSTM

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


sysfuncptr = duffingOscillatorODE
# sim time
t0, tf = 0, 20

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

lr = 0.001
lrSA = 0.001

input_size = degreesOfFreedom
output_size = degreesOfFreedom
num_layers = 1
hidden_size = 50
hidden_size = 30
p_dropout = 0.0
lookback = 4

'''
with a lookback window of 4, the input to the network is [t,t+1,t+2,t+3] and the output is [t+1,t+2,t+3,t+4]

for long time series, overlapping windows can be created.
A time series of n time steps can produce roughly n windows because a window can start from any time step as long as the window does 
not go beyond the boundary of the time series. Within one window, there are multiple consecutive time steps of values. 
In each time step, there can be multiple features.
'''

train_size = int(500)
test_size = len(output_seq) - train_size

train, test = output_seq[:train_size], output_seq[train_size:]

train_in,train_out = create_dataset(train,device,lookback=lookback)
test_in,test_out = create_dataset(test,device,lookback=lookback)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

# initilizing the model, criterion, and optimizer for the data
model = LSTM(input_size,hidden_size,output_size,num_layers, p_dropout).double().to(device)
modelSA = LSTMSelfAttentionNetwork(input_size,hidden_size,output_size,num_layers, p_dropout).double().to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
optimizerSA = torch.optim.Adam(modelSA.parameters(),lr=lrSA)

criterion = F.smooth_l1_loss
# criterion = torch.nn.MSELoss()
# Define the mean absolute error (MAE) loss function
# mae_loss = F.l1_loss(predicted, target)

# Define the Huber loss function with delta=1.0
# huber_loss = F.smooth_l1_loss(predicted, target, reduction='mean', delta=1.0)

def plotPredition(epoch,err=None,num = None):
        with torch.no_grad():
            # shift train predictions for plotting
            train_plot = np.ones_like(output_seq) * np.nan
            y_pred = model(train_in)
            y_pred = y_pred[:, -1, :]
            train_plot[lookback:train_size] = model(train_in)[:, -1, :].cpu()
            # shift test predictions for plotting
            test_plot = np.ones_like(output_seq) * np.nan
            test_plot[train_size+lookback:len(output_seq)] = model(test_in)[:, -1, :].cpu()

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

        plt.savefig('predict/predict%d.png' % epoch)
        plt.close()

        if err is not None:
            fig, (ax1, ax2) = plt.subplots(2,1)
            ax1.plot(np.linspace(0,tf,len(err)),err)
            ax1.set_title('Error btwn prediction and truth')
            ax1.set_ylabel('error')

            ax2.plot(np.linspace(0,tf,len(err)),np.average(err,axis=0)*np.ones(err.shape))
            ax2.set_ylabel('average error')
            ax2.set_xlabel('time (sec)')

            plt.legend(['Position Error','Velocity Error'],loc="best")

            if num == None:
                plt.savefig('predict/errorFinal_%ds_500pts.png' % tf)
            else:
                plt.savefig('predict/errorFinal_%ds_500pts%d.png' % (tf, num))

            plt.close()
        # filter out nan values for better post processing
        train_plot = train_plot[~np.isnan(train_plot)]
        test_plot = test_plot[~np.isnan(test_plot)]

        trajPredition = np.concatenate((train_plot,test_plot))

        return trajPredition.reshape((len(trajPredition),1))

for epoch in range(n_epochs):

    # plotPredition(epoch)

    model.train()
    modelSA.train()

    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred = modelSA(X_batch)
        lossSA = criterion(y_pred, y_batch)
        optimizerSA.zero_grad()
        lossSA.backward()
        optimizerSA.step()

    # Validation
    model.eval()
    modelSA.eval()

    with torch.no_grad():
        y_pred_train = model(train_in)
        train_loss = np.sqrt(criterion(y_pred_train, train_out).cpu())
        y_pred_test = model(test_in)
        test_loss = np.sqrt(criterion(y_pred_test, test_out).cpu())

        decAcc, err1 = findDecAcc(train_out,y_pred_train,printOut=False)
        decAcc, err2 = findDecAcc(test_out,y_pred_test)
        err = np.concatenate((err1,err2),axis=0)

    print('\nLSTM')
    print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))

    with torch.no_grad():
        y_pred_train = modelSA(train_in)
        train_loss = np.sqrt(criterion(y_pred_train, train_out).cpu())
        y_pred_test = modelSA(test_in)
        test_loss = np.sqrt(criterion(y_pred_test, test_out).cpu())

        decAcc, err1 = findDecAcc(train_out,y_pred_train,printOut=False)
        decAcc, err2 = findDecAcc(test_out,y_pred_test)
        errSA = np.concatenate((err1,err2),axis=0)
    
    print('\nLSTMSA')
    print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))


plotPredition(epoch+1,err = err,num=1)
plotPredition(epoch+1,err = errSA,num=2)

