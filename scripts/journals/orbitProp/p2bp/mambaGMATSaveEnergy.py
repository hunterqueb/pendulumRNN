
import torch
import torch.nn.functional as F
import os

from qutils.plot import plotStatePredictions
from qutils.orbital import readGMATReport, dim2NonDim6, nonDim2Dim6
from qutils.mamba import Mamba, MambaConfig
from qutils.ml import trainModel, getDevice, Adam_mini, create_datasets, LSTMSelfAttentionNetwork
from qutils.mlExtras import rmse
from matplotlib import pyplot as plt

compareLSTM = True
plotOn = False
printoutSuperweight = False


problemDim = 6

device = getDevice()

gmatImport = readGMATReport("gmat/data/reportHEO360Prop.txt")
semimajorAxis = 67903.82797675686
tPeriod = 175587.6732104912
# gmat propagation uses 50/70 50/70 JGM-2 with MSISE90 spherical drag model w/ SRP

t = gmatImport[:,-1]

output_seq = gmatImport[:,0:problemDim]

muR = 396800
DU = 6378.1 # radius of earth in km
TU = ((DU)**3 / muR)**0.5

output_seq = dim2NonDim6(output_seq,DU,TU)
print(output_seq[0,:])
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
p_motion_knowledge = 0.1


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

networkPrediction = plotStatePredictions(model,t,output_seq,train_in,test_in,train_size,test_size,DU=DU,TU=TU,plotOn=False)
output_seq = nonDim2Dim6(output_seq,DU,TU)

rmseMamba = rmse(output_seq,networkPrediction)

from qutils.orbital import orbitalEnergy

energyMamba = orbitalEnergy(networkPrediction)
energyMamba = energyMamba.reshape(len(energyMamba),1)

del model
del optimizer
torch.cuda.empty_cache()
import gc
gc.collect()
modelLSTM = returnModel('lstm')

optimizer = Adam_mini(modelLSTM,lr=lr)

timeToTrainLSTM = trainModel(modelLSTM,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutAcc = True,printOutToc = True)


output_seq = dim2NonDim6(output_seq,DU,TU)
networkPredictionLSTM = plotStatePredictions(modelLSTM,t,output_seq,train_in,test_in,train_size,test_size,DU=DU,TU=TU,plotOn=False)
output_seq = nonDim2Dim6(output_seq,DU,TU)

rmseLSTM = rmse(output_seq,networkPredictionLSTM)

energyLSTM = orbitalEnergy(networkPredictionLSTM)
energyLSTM = energyLSTM.reshape(len(energyLSTM),1)

energy = orbitalEnergy(output_seq)
energy = energy.reshape(len(energy),1)


import numpy as np

fieldnames = ["Mamba Energy","LSTM Energy"]
new_data = {"Mamba Energy":energyMamba,"LSTM Energy":energyLSTM}

file_path = 'p2bpEnergyMamba.npy'

try:
    # Load existing data and append new data
    existing_data = np.load(file_path)
    updated_data = np.hstack((existing_data, energyMamba))  # Append column-wise to make it 1000x100
except FileNotFoundError:
    # If the file doesn't exist, initialize the file with the first array
    updated_data = energyMamba

# Save the updated data back to the file
np.save(file_path, updated_data)


file_path = 'p2bpEnergyLSTM.npy'
try:
    # Load existing data and append new data
    existing_data = np.load(file_path)
    updated_data = np.hstack((existing_data, energyLSTM))  # Append column-wise to make it 1000x100
except FileNotFoundError:
    # If the file doesn't exist, initialize the file with the first array
    updated_data = energyLSTM

# Save the updated data back to the file
np.save(file_path, updated_data)

file_path = 'p2bpEnergy.npy'
try:
    # Load existing data and append new data
    existing_data = np.load(file_path)
    updated_data = np.hstack((existing_data, energy))  # Append column-wise to make it 1000x100
except FileNotFoundError:
    # If the file doesn't exist, initialize the file with the first array
    updated_data = energy

# Save the updated data back to the file
np.save(file_path, updated_data)