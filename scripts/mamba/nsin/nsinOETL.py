import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo

from qutils.integrators import ode87
from qutils.plot import plotStatePredictions,newPlotSolutionErrors
from qutils.mamba import Mamba, MambaConfig
from qutils.ml import trainModel, printModelParmSize, getDevice, Adam_mini, create_datasets, genPlotPrediction, transferMamba
from qutils.tictoc import timer
from scipy.io import loadmat,savemat

# from nets import Adam_mini

# from memory_profiler import profile

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

trainModel(model,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutAcc = True,printOutToc = True)

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

trainModel(newModel,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutAcc = True,printOutToc = True)

networkPredictionTarget = plotStatePredictions(model,t,OE_J2_drag,train_in,test_in,train_size,test_size,states = ('a','e','i','Omega','omega','f'),units=('km',' ','deg','deg','deg','deg'))
fig = plt.gcf()
fig.suptitle('Target System - J2 and Drag VLEO OE')

newPlotSolutionErrors(OE_J2_drag,networkPredictionTarget,t,states = ('a','e','i','Omega','omega','f'),units=('km',' ','deg','deg','deg','deg'))

errorAvg = np.nanmean(abs(networkPredictionTarget-OE_J2_drag), axis=0)
print("Average values of each dimension:")
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg}")


plt.show()