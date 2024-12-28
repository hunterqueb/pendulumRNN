import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo

from qutils.integrators import ode87
from qutils.plot import plot3dOrbitPredictions,plotOrbitPhasePredictions, plotSolutionErrors,plotPercentSolutionErrors, plotEnergy,plotStatePredictions,newPlotSolutionErrors
from qutils.mlExtras import findDecAcc
from qutils.orbital import nonDim2Dim6, returnCR3BPIC, readGMATReport, dim2NonDim6, orbitalEnergy
from qutils.mamba import Mamba, MambaConfig
from qutils.ml import printModelParmSize, getDevice, Adam_mini, create_datasets, genPlotPrediction, transferMamba
from qutils.tictoc import timer
from scipy.io import loadmat,savemat

# from nets import Adam_mini

# from memory_profiler import profile
from qutils.mlExtras import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight

problemDim = 6

device = getDevice()

fileLocation = './scripts/mamba/nsin/'
OE_file = "J2_plus_drag_sv_oe.mat"

matlabFile = loadmat(fileLocation+OE_file)

t = matlabFile["torbit"]
# source
OE_nominal_two_body = matlabFile["OE_nominal"]
# target
OE_J2_drag = matlabFile["OE_J2_drag"]

n_epochs = 5
lr = 0.01
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = 1
p_motion_knowledge = 0.5

train_size = int(len(OE_nominal_two_body) * p_motion_knowledge)
test_size = len(OE_nominal_two_body) - train_size

train_in,train_out,test_in,test_out = create_datasets(OE_nominal_two_body,1,train_size,device)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

config = MambaConfig(d_model=problemDim, n_layers=num_layers,d_conv=32)
model = Mamba(config).to(device).double()

optimizer = Adam_mini(model,lr=lr)

criterion = F.smooth_l1_loss

trainTime = timer()
for epoch in range(n_epochs):

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

trainTime.toc()

networkPredictionSource = plotStatePredictions(model,t,OE_nominal_two_body,train_in,test_in,train_size,test_size,states = ('a','e','i','Omega','omega','f'),units=('km',' ','deg','deg','deg','deg'))
fig = plt.gcf()
fig.suptitle('Source System - Nominal VLEO OE')

newPlotSolutionErrors(OE_nominal_two_body,networkPredictionSource,t,states = ('a','e','i','Omega','omega','f'),units=('km',' ','deg','deg','deg','deg'))

errorAvg = np.nanmean(abs(networkPredictionSource-OE_nominal_two_body), axis=0)
print("Average values of each dimension:")
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg}")









n_epochs = 3

newModel = Mamba(config).to(device).double()
newModel = transferMamba(model,newModel,[True,True,False])

train_in,train_out,test_in,test_out = create_datasets(OE_J2_drag,1,train_size,device)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

trainTime = timer()
for epoch in range(n_epochs):
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

    print("Epoch %d: train loss %f, test loss %f\n" % (epoch, train_loss, test_loss))
trainTime.toc()


networkPredictionTarget = plotStatePredictions(model,t,OE_J2_drag,train_in,test_in,train_size,test_size,states = ('a','e','i','Omega','omega','f'),units=('km',' ','deg','deg','deg','deg'))
fig = plt.gcf()
fig.suptitle('Target System - J2 and Drag VLEO OE')

newPlotSolutionErrors(OE_J2_drag,networkPredictionTarget,t,states = ('a','e','i','Omega','omega','f'),units=('km',' ','deg','deg','deg','deg'))

errorAvg = np.nanmean(abs(networkPredictionTarget-OE_J2_drag), axis=0)
print("Average values of each dimension:")
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg}")


plt.show()