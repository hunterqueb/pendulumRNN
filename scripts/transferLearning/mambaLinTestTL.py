'''
Tranfers learning with single layer mamba network. need to experiment more with freezing certain layers,
different dynamical systems
'''


import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo

from qutils.integrators import myRK4Py, ode45
from qutils.ml.utils import findDecAcc,generateTrajectoryPrediction

from nets import LSTMSelfAttentionNetwork, create_dataset, transferMamba

from qutils.ml.mamba import Mamba, MambaConfig

import control as ct
# pip install control

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")



# linear system for a simple harmonic oscillator
k = 1; m = 2
wr = np.sqrt(k/m)
F0 = 0.3
c = 0.1 # consider no damping for now
zeta = c / (2*m*wr)

def linPendulumODE(t,theta,u,p=[m,c,k]):
    dtheta1 = theta[1]
    dtheta2 = -k/m*(theta[0]) - (c/m) * theta[1]
    return np.array([dtheta1, dtheta2])

A = np.array(([0,1],[-k/m,-c/m]))

B = np.array([0,1])

C = np.eye(2)

D = 0

t0 = 0; tf = 100
dt = 0.01
t = np.linspace(t0,tf,int(tf/dt))

u = F0 * np.sin(t) # LTI system
# u = F0 * np.exp(-2*t) # LTV system

sys = ct.StateSpace(A,B,C,D)

results = ct.forced_response(sys,t,u,[1,0])
numericalResult = results.states.T

# plt.figure()
# plt.plot(t,results.states.T)
# plt.show()

n_epochs = 10
lr = 0.001
lookback = 1

nDim = 2
nLayers = 1
config = MambaConfig(nDim,nLayers)
# config = MambaConfig(nDim,nLayers,d_state = 2,expand_factor=4)

modelLSTM = LSTMSelfAttentionNetwork(2,20,2,1,0).double().to(device)
modelMamba = Mamba(config).to(device).double()

optimizer = torch.optim.Adam(modelMamba.parameters(),lr=lr)
criterion = F.smooth_l1_loss
# criterion = torch.nn.MSELoss()


# p_motion_knowledge = 0.02
# train_size = int(len(numericalResult) * p_motion_knowledge)
# test_size = len(numericalResult) - train_size

train_seconds = 2
train_size = int(dt * 10000 * train_seconds)
# train_size = 2
test_size = len(numericalResult) - train_size

train, test = numericalResult[:train_size], numericalResult[train_size:]

train_in,train_out = create_dataset(train,device,lookback=lookback)
test_in,test_out = create_dataset(test,device,lookback=lookback)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

def plotPredition(epoch,model):
        with torch.no_grad():
            # shift train predictions for plotting
            train_plot = np.ones_like(numericalResult) * np.nan
            y_pred = model(train_in)
            y_pred = y_pred[:, -1, :]
            train_plot[lookback:train_size] = model(train_in)[:, -1, :].cpu()
            # shift test predictions for plotting
            test_plot = np.ones_like(numericalResult) * np.nan
            test_plot[train_size+lookback:len(numericalResult)] = model(test_in)[:, -1, :].cpu()

        fig, (ax1, ax2) = plt.subplots(2,1)
        # plot
        ax1.plot(t,numericalResult[:,0], c='b',label = 'True Motion')
        ax1.plot(t,train_plot[:,0], c='r',label = 'Training Region')
        ax1.plot(t,test_plot[:,0], c='g',label = 'Predition')
        ax1.set_title('Mamba Solution to Forced Oscillator')
        # ax1.xlabel('time (sec)')
        ax1.set_ylabel('x (m)')
        # plt.legend(loc="lower left")

        ax2.plot(t,numericalResult[:,1], c='b',label = 'True Motion')
        ax2.plot(t,train_plot[:,1], c='r',label = 'Training Region')
        ax2.plot(t,test_plot[:,1], c='g',label = 'Predition')
        ax2.set_xlabel('time (sec)')
        ax2.set_ylabel('xdot (m/s)')
        plt.legend(loc="lower left")
        plt.show()
        plt.savefig('predict/predict%d.png' % epoch)
        plt.close()

        trajPredition = generateTrajectoryPrediction(train_plot,test_plot)

        return trajPredition


for epoch in range(n_epochs):
    modelMamba.train()
    for x_batch, y_batch in loader:
        y_pred = modelMamba(x_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    modelMamba.eval()
    with torch.no_grad():
        y_pred_train = modelMamba(train_in)
        train_loss = np.sqrt(criterion(y_pred_train, train_out).cpu())
        y_pred_test = modelMamba(test_in)
        test_loss = np.sqrt(criterion(y_pred_test, test_out).cpu())

        decAcc, err1 = findDecAcc(train_out,y_pred_train,printOut=False)
        decAcc, err2 = findDecAcc(test_out,y_pred_test)
        err = np.concatenate((err1,err2),axis=0)

    print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))

# if model == modelMamba:
#     print(model.layers[0].mixer.A_SSM.shape)
#     print(model.layers[0].mixer.B_SSM.shape)
#     print(model.layers[0].mixer.C_SSM.shape)
#     print(model.layers[0].mixer.delta)
# A takes the a shape defined by the user, a combination of the user defined latent space size and the expansion size of the input
# B and C take the size of the test vector? how is it doing this? how does it now
torchinfo.summary(modelMamba)

plotPredition(n_epochs,modelMamba)


# TRANSFER LEARNING


# linear system for a simple harmonic oscillator
k = 2; m = 1
wr = np.sqrt(k/m)
F0 = .1
c = 0.1 # consider no damping for now
zeta = c / (2*m*wr)

def linPendulumODE(t,theta,u,p=[m,c,k]):
    dtheta1 = theta[1]
    dtheta2 = -k/m*(theta[0]) - (c/m) * theta[1]
    return np.array([dtheta1, dtheta2])

A = np.array(([0,1],[-k/m,-c/m]))

B = np.array([0,1])

C = np.eye(2)

D = 0

u = F0 * np.sin(t) # LTI system
# u = F0 * np.exp(-2*t) # LTV system

sys = ct.StateSpace(A,B,C,D)

newIC = np.random.rand(2, 1)

results = ct.forced_response(sys,t,u,newIC)
numericalResult = results.states.T

# plt.figure()
# plt.plot(t,results.states.T)
# plt.show()

n_epochs = 10
lr = 0.001
lookback = 1

nDim = 2
nLayers = 1

modelLSTM = LSTMSelfAttentionNetwork(2,20,2,1,0).double().to(device)
newModelMamba = Mamba(config).to(device).double()


newModelMamba = transferMamba(modelMamba,newModelMamba)

# model = modelLSTM

optimizer = torch.optim.Adam(newModelMamba.parameters(),lr=lr)
criterion = F.smooth_l1_loss
# criterion = torch.nn.MSELoss()


# p_motion_knowledge = 0.02
# train_size = int(len(numericalResult) * p_motion_knowledge)
# test_size = len(numericalResult) - train_size

# train_size = int(dt * 10000 * 2)
# train_size = 2
test_size = len(numericalResult) - train_size

train, test = numericalResult[:train_size], numericalResult[train_size:]

train_in,train_out = create_dataset(train,device,lookback=lookback)
test_in,test_out = create_dataset(test,device,lookback=lookback)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

for epoch in range(n_epochs):
    newModelMamba.train()
    for x_batch, y_batch in loader:
        y_pred = newModelMamba(x_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    newModelMamba.eval()
    with torch.no_grad():
        y_pred_train = newModelMamba(train_in)
        train_loss = np.sqrt(criterion(y_pred_train, train_out).cpu())
        y_pred_test = newModelMamba(test_in)
        test_loss = np.sqrt(criterion(y_pred_test, test_out).cpu())

        decAcc, err1 = findDecAcc(train_out,y_pred_train,printOut=False)
        decAcc, err2 = findDecAcc(test_out,y_pred_test)
        err = np.concatenate((err1,err2),axis=0)

    print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))

torchinfo.summary(newModelMamba)
plotPredition(n_epochs,newModelMamba)
