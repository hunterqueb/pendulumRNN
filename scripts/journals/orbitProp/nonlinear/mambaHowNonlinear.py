import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import torch
import torch.nn.functional as F
import torchinfo
import sys

from qutils.integrators import myRK4
from qutils.plot import plotCR3BPPhasePredictions,plotOrbitPredictions, plotSolutionErrors,plot3dCR3BPPredictions,plotStatePredictions, plotEnergy
from qutils.mlExtras import findDecAcc,printoutMaxLayerWeight
from qutils.orbital import nonDim2Dim6, dim2NonDim6, returnCR3BPIC, jacobiConstant6
from qutils.mamba import Mamba, MambaConfig
from qutils.ml import trainModel, printModelParmSize, getDevice, Adam_mini, genPlotPrediction, create_datasets,LSTMSelfAttentionNetwork
from qutils.tictoc import timer
# from nets import Adam_mini

# from memory_profiler import profile
from qutils.mlExtras import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight
from qutils.mlSuperweight import findMambaSuperActivation, plotSuperActivation

if len(sys.argv) > 1:
    learnStable = True if sys.argv[1] == 1 else False
else:
    learnStable = True

plotOn = True
printoutSuperweight = True
saveSuperweightToCSV = False

nSamples = 100

problemDim = 3

from numpy import cos,sin

def nominalEulerAngleMotion(t, x,p=None):  

    s3t = sin(3*t)
    c3t = cos(3*t)
    s5t = sin(5*t)
    c5t = cos(5*t)

    dydt1 = 3*c3t*c5t - 5*s3t*s5t
    dydt2 = 5.5*c5t
    dydt3 = 0.5 * (c5t*5*(0.1+s3t)**3 + 9*c3t*(0.1+s3t)**2*s5t)
    # dydt3 = 0.5 * 9 * s5t*c3t*(s3t+0.1)**2

    return np.array([dydt1, dydt2,dydt3])

np.random.seed()

def uniformRandomPointOnSphere():
    """Generates a random point on the surface of a unit sphere."""

    radius = (10/180)

    # Generate a random point within a cube
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    z = np.random.uniform(-1, 1)

    # Normalize the point to project it onto the sphere
    norm = np.sqrt(x**2 + y**2 + z**2)
    while norm > 1:  # Ensure the point is inside the sphere
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(-1, 1)
        norm = np.sqrt(x**2 + y**2 + z**2)

    x = x / norm * radius
    y = y / norm * radius
    z = z / norm * radius

    return x, y, z

# dphi, dtheta, dpsi

IC = np.array((uniformRandomPointOnSphere()))

# IC = np.zeros((3,))


device = getDevice()



t0 = 0; tf = 25

# delT = 0.001
# nSamples = int(np.ceil((tf - t0) / delT))
t = np.linspace(t0, tf, nSamples)

system = nominalEulerAngleMotion

numericResult = myRK4(system,IC,t,paramaters=None)

plt.figure(figsize=(10,3))
plt.plot(t,numericResult[:,0])
plt.ylim((-1,1))

plt.figure(figsize=(10,3))
plt.plot(t,numericResult[:,1])
plt.ylim((-2,2))

plt.figure(figsize=(10,3))
plt.plot(t,numericResult[:,2])
plt.ylim((-1,1))


output_seq = numericResult

# hyperparameters
n_epochs = 5
# lr = 5*(10**-5)
# lr = 0.85
lr = 0.8
lr = 0.08
lr = 0.004
lr = 0.001
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = 1
p_motion_knowledge = 0.5


train_size = int(len(output_seq) * p_motion_knowledge)
# train_size = 2
test_size = len(output_seq) - train_size

train_in,train_out,test_in,test_out = create_datasets(output_seq,1,train_size,device)

# initilizing the model, criterion, and optimizer for the data
config = MambaConfig(d_model=problemDim, n_layers=num_layers,d_conv=16)

def returnModel(modelString = 'mamba'):
    if modelString == 'mamba':
        model = Mamba(config).to(device).double()
    elif modelString == 'lstm':
        model = LSTMSelfAttentionNetwork(input_size,30,output_size,num_layers,0).double().to(device)
    return model

model = returnModel()

optimizer = Adam_mini(model,lr=lr)
# optimizer = Adam_mini(model,lr=lr)

criterion = F.smooth_l1_loss
# criterion = torch.nn.HuberLoss()

trainModel(model,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutAcc = True,printOutToc = True)

networkPrediction = plotStatePredictions(model,t,output_seq,train_in,test_in,train_size,test_size,states=("None","None","None"),units=("None","None","None"))

from qutils.plot import newPlotSolutionErrors
newPlotSolutionErrors(output_seq,networkPrediction,t,states=("None","None","None"),units=("None","None","None"))

from qutils.mlExtras import rmse

rmse(output_seq,networkPrediction)

errorAvg = np.nanmean(abs(networkPrediction-output_seq), axis=0)
print("Average values of each dimension:")
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg}")

printModelParmSize(model)
torchinfo.summary(model)
print('rk85 on 2 period halo orbit takes 1.199 MB of memory to solve')
print(numericResult[0,:])
print(numericResult[1,:])

magnitude, index = findMambaSuperActivation(model,test_in)
normedMags = np.zeros((len(magnitude),))
for i in range(len(magnitude)):
    normedMags[i] = magnitude[i].norm().detach().cpu()


if printoutSuperweight is True:
    printoutMaxLayerWeight(model)
    getSuperWeight(model)
    plotSuperWeight(model)
    plotSuperActivation(magnitude, index,printOutValues=True)


if saveSuperweightToCSV is True:
    import csv
    import os
    fieldnames = ["in_proj","conv1d","x_proj","dt_proj","out_proj"]
    new_data = {"in_proj":normedMags[0],"conv1d":normedMags[1],"x_proj":normedMags[2],"dt_proj":normedMags[3],"out_proj":normedMags[4]}


    file_path = 'superWeight' + str(nSamples) + 'Samples.csv'
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file,fieldnames=fieldnames)
        if not file_exists or os.path.getsize(file_path) == 0:
            writer.writeheader()
        writer.writerow(new_data)


if plotOn is True:
    plt.show()
