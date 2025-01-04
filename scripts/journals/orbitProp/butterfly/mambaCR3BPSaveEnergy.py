
import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt

from qutils.integrators import ode87
from qutils.plot import plotStatePredictions
from qutils.orbital import dim2NonDim6, returnCR3BPIC,nonDim2Dim6
from qutils.mamba import Mamba, MambaConfig
from qutils.ml import trainModel, getDevice, Adam_mini, create_datasets, LSTMSelfAttentionNetwork
from qutils.mlExtras import rmse
compareLSTM = True
plotOn = False
printoutSuperweight = False


problemDim = 6

device = getDevice()

orbitFamily = 'butterfly'

CR3BPIC = returnCR3BPIC(orbitFamily,id=1080)
x_0,tEnd = CR3BPIC()

problemDim = 6
m_1 = 5.974E24  # kg
m_2 = 7.348E22 # kg
mu = m_2/(m_1 + m_2)


IC = np.array(x_0)

def system(t, Y,mu=mu):
    """Solve the CR3BP in nondimensional coordinates.
    
    The state vector is Y, with the first three components as the
    position of $m$, and the second three components its velocity.
    
    The solution is parameterized on $\\pi_2$, the mass ratio.
    """
    # Get the position and velocity from the solution vector
    x, y, z = Y[:3]
    xdot, ydot, zdot = Y[3:]

    # Define the derivative vector

    dydt1 = xdot
    dydt2 = ydot
    dydt3 = zdot

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)

    dydt4 = 2 * ydot + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    dydt5 = -2 * xdot + y - (1 - mu) * y / r1**3 - mu * y / r2**3
    dydt6 = -(1 - mu) * z / r1**3 - mu * z / r2**3

    return np.array([dydt1, dydt2,dydt3,dydt4,dydt5,dydt6])



device = getDevice()

numPeriods = 4

t0 = 0; tf = numPeriods * tEnd

delT = 0.001
nSamples = int(np.ceil((tf - t0) / delT))
t = np.linspace(t0, tf, nSamples)

from scipy.io import loadmat
cr3bp = loadmat("scripts/journals/orbitProp/butterfly/CR3BP_butterfly_1080.mat")

output_seq = cr3bp['XODE']
t = cr3bp['tResult']

t = t / tEnd


muR = 396800
DU = 389703
G = 6.67430e-11
# TU = np.sqrt(DU**3 / (G*(m_1+m_2)))
TU = 382981

# hyperparameters
n_epochs = 5
# lr = 5*(10**-5)
# lr = 0.85
lr = 0.8
lr = 0.08
lr = 0.004
lr = 0.001
lr = 0.01
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = 1
p_motion_knowledge = 1/numPeriods


train_size = int(len(output_seq) * p_motion_knowledge)
# train_size = 2
test_size = len(output_seq) - train_size
print(train_size)
print(test_size)
train_in,train_out,test_in,test_out = create_datasets(output_seq,1,train_size,device)

# initilizing the model, criterion, and optimizer for the data
config = MambaConfig(d_model=problemDim, n_layers=num_layers,d_conv=32)

def returnModel(modelString = 'mamba'):
    if modelString == 'mamba':
        model = Mamba(config).to(device).double()
    elif modelString == 'lstm':
        model = LSTMSelfAttentionNetwork(input_size,30,output_size,num_layers,0).double().to(device)
    return model

model = returnModel()

# optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
optimizer = Adam_mini(model,lr=lr)

criterion = F.smooth_l1_loss
# criterion = torch.nn.HuberLoss()
# criterion = F.mse_loss

timeToTrain = trainModel(model,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutAcc = True,printOutToc = True)

networkPrediction = plotStatePredictions(model,t,output_seq,train_in,test_in,train_size,test_size,plotOn=True)

plt.show()
del model
del optimizer
torch.cuda.empty_cache()
import gc
gc.collect()
modelLSTM = returnModel('lstm')

optimizer = Adam_mini(modelLSTM,lr=lr)

timeToTrainLSTM = trainModel(modelLSTM,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutAcc = True,printOutToc = True)


networkPredictionLSTM = plotStatePredictions(modelLSTM,t,output_seq,train_in,test_in,train_size,test_size,plotOn=False)

from qutils.orbital import jacobiConstant6

energyMamba = jacobiConstant6(networkPrediction)
energyLSTM = jacobiConstant6(networkPredictionLSTM)

plt.plot(t,energyMamba)
plt.plot(t,energyLSTM)

plt.show()
import numpy as np

fieldnames = ["Mamba Energy","LSTM Energy"]
new_data = {"Mamba Energy":energyMamba,"LSTM Energy":energyLSTM}

file_path = 'cr3bpEnergyMamba.npy'

try:
    # Load existing data and append new data
    existing_data = np.load(file_path)
    updated_data = np.hstack((existing_data, energyMamba))  # Append column-wise to make it 1000x100
except FileNotFoundError:
    # If the file doesn't exist, initialize the file with the first array
    updated_data = energyMamba

# Save the updated data back to the file
np.save(file_path, updated_data)


file_path = 'cr3bpEnergyLSTM.npy'
try:
    # Load existing data and append new data
    existing_data = np.load(file_path)
    updated_data = np.hstack((existing_data, energyLSTM))  # Append column-wise to make it 1000x100
except FileNotFoundError:
    # If the file doesn't exist, initialize the file with the first array
    updated_data = energyLSTM

# Save the updated data back to the file
np.save(file_path, updated_data)
