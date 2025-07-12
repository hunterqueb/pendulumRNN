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
from qutils.plot import plotOrbitPhasePredictions,plotSolutionErrors
from qutils.orbital import nonDim2Dim4
from qutils.tictoc import timer
from qutils.ml.mamba import Mamba, MambaConfig
from qutils.ml import getDevice

from nets import LSTMSelfAttentionNetwork, create_dataset

import control as ct
# pip install control

device = getDevice()


# linear system for a simple harmonic oscillator
k = 1; m = 1
wr = np.sqrt(k/m)
F0 = 0.3
c = 0.1 # consider no damping for now
zeta = c / (2*m*wr)

def linPendulumODE(t,theta,p=[m,c,k]):
    dtheta1 = theta[1]
    dtheta2 = -k/m*(theta[0]) - (c/m) * theta[1]
    return np.array([dtheta1, dtheta2])


t0 = 0; tf = 10
dt = 0.01
t = np.linspace(t0,tf,int(tf/dt))

# results = ct.forced_response(sys,t,u,[1,0])
# numericalResult = results.states.T

# plt.figure()
# plt.plot(t,results.states.T)
# plt.show()

n_epochs = 5000
lr = 0.0001
lookback = 1

nDim = 2
nLayers = 1
config = MambaConfig(nDim,nLayers,expand_factor=4)

modelMamba = Mamba(config).to(device).double()

model = modelMamba
# model = modelLSTM
torchinfo.summary(model)

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = F.smooth_l1_loss
# criterion = torch.nn.MSELoss()


totalData = 500

stateData = []
for i in range(totalData):
    IC = np.random.uniform(-1, 1, (nDim))
    _ , y = ode45(linPendulumODE,(t0,tf),IC,t)
    stateData.append(y)
stateData = np.array(stateData)

# TODO i dont think this is right

p_motion_knowledge = 0.80
train_size = int(totalData * p_motion_knowledge)
test_size = totalData - train_size

# train_size = int(dt * 10000 * 2)
# # train_size = 2
# test_size = len(numericalResult) - train_size

train, test = stateData[:train_size], stateData[train_size:]

train_in,train_out = create_dataset(train,device,lookback=lookback)
test_in,test_out = create_dataset(test,device,lookback=lookback)

train_in = train_in.squeeze(1)
train_out= train_out.squeeze(1)
test_in  = test_in.squeeze(1)
test_out  = test_out.squeeze(1)


loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

def plotPredition(epoch):
        '''
        prediction changes to 
        # i have an ic of shape 1,1,2
        # get out data from this
        # put out data back into itself
        '''
        with torch.no_grad():
            # shift train predictions for plotting
            train_plot = np.ones_like(stateData) * np.nan
            y_pred = model(train_in)
            y_pred = y_pred[:, -1, :]
            train_plot[lookback:train_size] = model(train_in)[:, -1, :].cpu()
            # shift test predictions for plotting
            test_plot = np.ones_like(stateData) * np.nan
            test_plot[train_size+lookback:len(stateData)] = model(test_in)[:, -1, :].cpu()

        fig, (ax1, ax2) = plt.subplots(2,1)
        # plot
        ax1.plot(t,stateData[:,0], c='b',label = 'True Motion')
        ax1.plot(t,train_plot[:,0], c='r',label = 'Training Region')
        ax1.plot(t,test_plot[:,0], c='g',label = 'Predition')
        ax1.set_title('Mamba Solution to Linear Oscillator')
        # ax1.xlabel('time (sec)')
        ax1.set_ylabel('x (m)')
        # plt.legend(loc="lower left")

        ax2.plot(t,stateData[:,1], c='b',label = 'True Motion')
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


initialConditions = np.random.uniform(-1, 1, (1, 1, nDim))
odeIC = np.squeeze(initialConditions)

_ , y_truth = ode45(linPendulumODE,(t0,tf),odeIC,t)

data_in = torch.tensor(initialConditions,device=device)
predictionTime = timer()
data_out_array = np.empty((1, int(tf/dt), nDim))

for i in range(int(tf/dt)):
    with torch.no_grad():
        data_out = model(data_in)
        data_out_array[0,i,:] = data_out.cpu().numpy()
        data_in = data_out
predictionTime.toc()
y_prediction = np.squeeze(data_out_array)


plotSolutionErrors(y_truth,y_prediction,t)
# plotPredition(n_epochs)

plt.savefig('finalError.png')