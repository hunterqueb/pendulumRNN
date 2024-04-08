'''

i dont know what this script does, but im leaving it for now just in case. i think its for unsupervised learning
maybe just a test file

'''


import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import random
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo

from qutils.integrators import myRK4Py, ode45
from qutils.mlExtras import findDecAcc
from qutils.plot import plotOrbitPhasePredictions
from qutils.orbital import nonDim2Dim4

from nets import LSTMSelfAttentionNetwork, create_dataset

from qutils.mamba import Mamba, MambaConfig

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
k = 1; m = 1
wr = np.sqrt(k/m)
F0 = 0.1
c = 0.6 # consider no damping for now
zeta = c / (2*m*wr)

def linPendulumODE(t,theta,u,p=[m,c,k]):
    dtheta1 = theta[1]
    dtheta2 = -k/m*(theta[0]) - (c/m) * theta[1]
    return np.array([dtheta1, dtheta2])

A = np.array(([0,1],[-k/m,-c/m]))

B = np.array([0,1])

C = np.array(([1,0],[0,0]))

D = 0

t0 = 0; tf = 10
dt = 0.001
t = np.linspace(t0,tf,int(tf/dt))

u = F0 * np.sin(t)

sys = ct.StateSpace(A,B,C,D)

results = ct.forced_response(sys,t,u,[1,0])
numericalResult = results.states.T[:,0]
u = results.u.T

# plt.figure()
# plt.plot(t,results.states.T)
# plt.show()

n_epochs = 1
lr = 0.001
lookback = 1
p_motion_knowledge = 0.02

nDim = 2
nLayers = 1
config = MambaConfig(nDim,nLayers,d_state = 2,expand_factor=1)

modelLSTM = LSTMSelfAttentionNetwork(2,20,2,1, 0).double().to(device)
modelMamba = Mamba(config).to(device).double()

model = modelMamba
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = F.smooth_l1_loss


train_size = int(len(u) * p_motion_knowledge)
test_size = len(u) - train_size

train, test = u[:train_size], u[train_size:]

train_in,train_out = create_dataset(train,device,lookback=lookback)
test_in,test_out = create_dataset(test,device,lookback=lookback)

train_size = int(len(numericalResult) * p_motion_knowledge)
test_size = len(numericalResult) - train_size

train, test = numericalResult[:train_size], numericalResult[train_size:]

_,train_out = create_dataset(train,device,lookback=lookback)
_,test_out = create_dataset(test,device,lookback=lookback)

train_out = train_out.reshape(train_in.shape)
test_out = test_out.reshape(test_in.shape)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)


def plotPredition(epoch):
        with torch.no_grad():
            # shift train predictions for plotting
            train_plot = np.ones_like(numericalResult) * np.nan
            y_pred = model(train_in)
            train_plot[lookback:train_size] = model(train_in).cpu()
            # shift test predictions for plotting
            test_plot = np.ones_like(numericalResult) * np.nan
            test_plot[train_size+lookback:len(numericalResult)] = model(test_in)[:, -1, :].cpu()

        fig, (ax1, ax2) = plt.subplots(2,1)
        # plot
        ax1.plot(t,numericalResult[:,0], c='b',label = 'True Motion')
        ax1.plot(t,train_plot[:,0], c='r',label = 'Training Region')
        ax1.plot(t,test_plot[:,0], c='g',label = 'Predition')
        ax1.set_title('Mamba Solution to Linear Oscillator')
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

        # filter out nan values for better post processing
        train_plot = train_plot[~np.isnan(train_plot)]
        test_plot = test_plot[~np.isnan(test_plot)]

        trajPredition = np.concatenate((train_plot,test_plot))

        return trajPredition.reshape((len(trajPredition),1))


for epoch in range(n_epochs):
    model.train()
    for x_batch, y_batch in loader:
        y_pred = model(x_batch)
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
        # err = np.concatenate((err1,err2),axis=0)

    print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))

if model == modelMamba:
    print(model.layers[0].mixer.A_SSM.shape)
    print(model.layers[0].mixer.B_SSM.shape)
    print(model.layers[0].mixer.C_SSM.shape)
#     print(model.layers[0].mixer.delta)

# A takes the a shape defined by the user, a combination of the user defined latent space size and the expansion size of the input
# B and C take the size of the test vector? how is it doing this? how does it now
plotPredition(n_epochs)