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
from qutils.mlExtras import rmse
from qutils.tictoc import timer
from scipy.io import loadmat,savemat

# from nets import Adam_mini

# from memory_profiler import profile

problemDim = 7
problemStates = ('a','e','i','Omega','omega','f','m')
problemUnits = ('km',' ','deg','deg','deg','deg','kg')
device = getDevice()

fileLocation = './scripts/mamba/nsin/'
OE_file = "orbitRaisingOE.mat"

matlabFile = loadmat(fileLocation+OE_file)

t = matlabFile["torbit"]

# # source
# OE_nominal_J2_drag = matlabFile["X_nominal_J2_drag"]
# # target
# OE_J2_drag_thrust = matlabFile["X_nominal_thrust"]


def normByDim(array):
    normVect = []
    newArray = np.zeros_like(array)
    for i in range(len(array[-1,:])):
        normFactor = max(array[:,i])
        newArray[:,i] = array[:,i] / normFactor
        normVect.append(normFactor)
    return newArray, normVect

def unNormByDim(array,normVect):
    newArray = np.zeros_like(array)
    for i in range(len(array[-1,:])):
        normFactor = normVect[i]
        newArray[:,i] = array[:,i] * normFactor
    return newArray


# def normByDim(array):
#     normVect = []
#     for i in range(len(array[-1,:])):
#         normVect.append(1)
#     return array, normVect

# def unNormByDim(array,normVect):
#     return array


# source
OE_nominal_J2_drag = matlabFile["OE_nominal_J2_drag_mass"]
# target
OE_J2_drag_thrust = matlabFile["OE_nominal_thrust_mass"]

OE_nominal_J2_drag, nomNormVect = normByDim(OE_nominal_J2_drag)
OE_J2_drag_thrust, thrustNormVect = normByDim(OE_J2_drag_thrust)


n_epochs = 5
lr = 0.01
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = 1
p_motion_knowledge = 0.5

train_size = int(len(OE_nominal_J2_drag) * p_motion_knowledge)
test_size = len(OE_nominal_J2_drag) - train_size

train_in,train_out,test_in,test_out = create_datasets(OE_nominal_J2_drag,1,train_size,device)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

config = MambaConfig(d_model=problemDim, n_layers=num_layers,d_conv=32)
model = Mamba(config).to(device).double()

optimizer = Adam_mini(model,lr=lr)

criterion = F.smooth_l1_loss

trainModel(model,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutAcc = True,printOutToc = True)

networkPredictionSource = plotStatePredictions(model,t,OE_nominal_J2_drag,train_in,test_in,train_size,test_size,states = problemStates,units=problemUnits)
fig = plt.gcf()
fig.suptitle('Source System - J2 and Drag VLEO OE')

OE_nominal_J2_drag = unNormByDim(OE_nominal_J2_drag,nomNormVect)
networkPredictionSource = unNormByDim(networkPredictionSource,nomNormVect)

newPlotSolutionErrors(OE_nominal_J2_drag,networkPredictionSource,t,states = problemStates,units=problemUnits)

errorAvg = np.nanmean(abs(networkPredictionSource-OE_nominal_J2_drag), axis=0)
print("Average values of each dimension:")
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg}")



n_epochs = 3

newModel = Mamba(config).to(device).double()
newModel = transferMamba(model,newModel,[True,True,False])

train_in,train_out,test_in,test_out = create_datasets(OE_J2_drag_thrust,1,train_size,device)

trainModel(newModel,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutAcc = True,printOutToc = True)

networkPredictionTarget = plotStatePredictions(model,t,OE_J2_drag_thrust,train_in,test_in,train_size,test_size,states = problemStates,units=problemUnits)
fig = plt.gcf()
fig.suptitle('Target System - J2, Drag, and Thrust VLEO OE')


OE_J2_drag_thrust = unNormByDim(OE_J2_drag_thrust,thrustNormVect)
networkPredictionTarget = unNormByDim(networkPredictionTarget,thrustNormVect)


newPlotSolutionErrors(OE_J2_drag_thrust,networkPredictionTarget,t,states = problemStates,units=problemUnits)


errorAvg = np.nanmean(abs(networkPredictionTarget-OE_J2_drag_thrust), axis=0)
print("Average values of each dimension:")
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg}")


OE_J2_drag_thrust, thrustNormVect = normByDim(OE_J2_drag_thrust)


n_epochs = 3

modelnoTL = Mamba(config).to(device).double()

train_in,train_out,test_in,test_out = create_datasets(OE_J2_drag_thrust,1,train_size,device)

trainModel(modelnoTL,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutAcc = True,printOutToc = True)


networkPredictionTargetnoTL = plotStatePredictions(modelnoTL,t,OE_J2_drag_thrust,train_in,test_in,train_size,test_size,states = problemStates,units=problemUnits)
fig = plt.gcf()
fig.suptitle('Target System No Transfer Learning - J2, Drag, and Thrust VLEO OE')

OE_J2_drag_thrust = unNormByDim(OE_J2_drag_thrust,thrustNormVect)
networkPredictionTargetnoTL = unNormByDim(networkPredictionTargetnoTL,thrustNormVect)

newPlotSolutionErrors(OE_J2_drag_thrust,networkPredictionTargetnoTL,t,states = problemStates,units=problemUnits)

errorAvg = np.nanmean(abs(networkPredictionTarget-OE_J2_drag_thrust), axis=0)
print("Average values of each dimension:")
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg}")


print('Source (J2 + drag) [5 training epochs]')
rmse(OE_nominal_J2_drag,networkPredictionSource)

print('Target w/ TL (J2 + drag + thrust) [3 training epochs]')
rmse(OE_J2_drag_thrust,networkPredictionTarget)

print('Target w/out TL (J2 + drag + thrust) [3 training epochs]')
rmse(OE_J2_drag_thrust,networkPredictionTargetnoTL)


plt.show()