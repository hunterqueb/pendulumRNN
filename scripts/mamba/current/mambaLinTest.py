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
from qutils.plot import plotOrbitPhasePredictions,newPlotSolutionErrors,plotStatePredictions
from qutils.orbital import nonDim2Dim4
from qutils.ml import create_datasets, LSTMSelfAttentionNetwork

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
F0 = 0.3
c = 0.0 # consider no damping for now
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
# u = F0
sys = ct.StateSpace(A,B,C,D)

results = ct.forced_response(sys,t,u,[1,0])
numericalResult = results.states.T

results = ct.forced_response(sys,t,0,[1,0])
numericalResultUnforced = results.states.T


# plt.figure()
# plt.plot(t,results.states.T)
# plt.show()

n_epochs = 50
lr = 0.001
lookback = 1

nDim = 2
nLayers = 1
config = MambaConfig(nDim,nLayers,d_state = 2,expand_factor=4)

modelLSTM = LSTMSelfAttentionNetwork(2,20,2,1,0).double().to(device)
modelMamba = Mamba(config).to(device).double()

model = modelMamba
# model = modelLSTM

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = F.smooth_l1_loss
# criterion = torch.nn.MSELoss()


# p_motion_knowledge = 0.02
# train_size = int(len(numericalResult) * p_motion_knowledge)
# test_size = len(numericalResult) - train_size

train_size = int(dt * 10000 * 2)
# train_size = 2
test_size = len(numericalResult) - train_size

train, test = numericalResult[:train_size], numericalResult[train_size:]

train_in,train_out,test_in,test_out = create_datasets(numericalResult,1,train_size,device)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

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
        err = np.concatenate((err1,err2),axis=0)

    print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))

if model == modelMamba:
    print(model.layers[0].mixer.A_SSM.shape)
    print(model.layers[0].mixer.B_SSM.shape)
    print(model.layers[0].mixer.C_SSM.shape)
    print(model.layers[0].mixer.delta)
# A takes the a shape defined by the user, a combination of the user defined latent space size and the expansion size of the input
# B and C take the size of the test vector? how is it doing this? how does it now
torchinfo.summary(model)

trajPredition = plotStatePredictions(model,t,numericalResult,train_in,test_in,train_size,test_size,states=['x','y','z'])
fig, axes = plt.subplots(2,1)
for i, ax in enumerate(axes.flat):
    ax.plot(t, numericalResultUnforced[:, i], c='b', label='Unforced Motion')

plt.legend()
newPlotSolutionErrors(numericalResult,trajPredition,t,states=['x','y','z'])
plt.show()
