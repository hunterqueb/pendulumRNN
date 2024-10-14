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

from qutils.mamba import Mamba, MambaConfig
from qutils.ml import create_datasets,genPlotPrediction
from qutils.tf import estimateTranferFunction,estimateMIMOsys
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

t0 = 0; tf = 10
dt = 0.01
t = np.linspace(t0,tf,int(tf/dt))

u = F0 * np.sin(t) # LTI system
u = F0 # LTI system
# u=0
# u = F0 * np.exp(-2*t) # LTV system

sys = ct.StateSpace(A,B,C,D)

results = ct.forced_response(sys,t,u,[1,0])
numericalResult = results.states.T
IC = numericalResult[0]
# plt.figure()
# plt.plot(t,results.states.T)
# plt.show()

n_epochs = 50
lr = 0.001
lookback = 1

nDim = 2
problemDim = nDim
nLayers = 1
config = MambaConfig(nDim,nLayers,d_state = 2,expand_factor=4)

modelMamba = Mamba(config).to(device).double()

model = modelMamba

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

def plotPredition(epoch):
        train_plot, test_plot = genPlotPrediction(model,numericalResult,train_in,test_in,train_size,1)

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
estimateMIMOsys(model,device,IC,t)
# estimateTranferFunction(model,device,IC,t)

plotPredition(n_epochs)

