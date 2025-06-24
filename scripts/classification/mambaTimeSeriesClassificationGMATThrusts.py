import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from qutils.tictoc import timer

from qutils.ml import getDevice, trainClassifier, LSTMClassifier, MambaClassifier, printModelParmSize, validateMultiClassClassifier
from qutils.mamba import Mamba, MambaConfig
from qutils.mlExtras import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight
from qutils.mlSuperweight import findMambaSuperActivation,plotSuperActivation,zeroModelWeight

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no-lstm',dest="use_lstm", action='store_false', help='Use LSTM model')
parser.add_argument("--systems", type=int, default=10000, help="Number of random systems to access")
parser.add_argument("--propMin", type=int, default=30, help="Minimum propagation time in minutes")
parser.add_argument("--orbit", type=str, default="vleo", help="Orbit type: vleo, leo")
parser.set_defaults(use_lstm=True)

args = parser.parse_args()
use_lstm = args.use_lstm
numMinProp = args.propMin
numRandSys = args.systems
orbitType = args.orbit

dataLoc = "gmat/data/classification/"+ orbitType +"/" + str(numMinProp) + "min-" + str(numRandSys)

# get npz files in folder and load them into script

a = np.load(f"{dataLoc}/statesArrayChemical.npz")
statesArrayChemical = a['statesArrayChemical']

a = np.load(f"{dataLoc}/statesArrayElectric.npz")
statesArrayElectric = a['statesArrayElectric']

a = np.load(f"{dataLoc}/statesArrayImpBurn.npz")
statesArrayImpBurn = a['statesArrayImpBurn']

a = np.load(f"{dataLoc}/statesArrayNoThrust.npz")
statesArrayNoThrust = a['statesArrayNoThrust']

del a

device = getDevice()

batchSize = 16
problemDim = 6

# Hyperparameters
input_size = problemDim 
hidden_size = 48 # must be multiple of train dim
num_layers = 2
num_classes = 4  # e.g., multiclass classification
learning_rate = 1e-3
num_epochs = 100

config = MambaConfig(d_model=input_size,n_layers = num_layers,expand_factor=hidden_size//input_size,d_state=32,d_conv=16,classifer=True)

noThrustLabel = 0
chemicalLabel = 1
electricLabel = 2
impBurnLabel = 3

# Create labels for each dataset
labelsChemical = np.full((statesArrayChemical.shape[0],1),chemicalLabel)
labelsElectric = np.full((statesArrayElectric.shape[0],1),electricLabel)
labelsImpBurn = np.full((statesArrayImpBurn.shape[0],1),impBurnLabel)
labelsNoThrust = np.full((statesArrayNoThrust.shape[0],1),noThrustLabel)
# Combine datasets and labels
dataset = np.concatenate((statesArrayChemical, statesArrayElectric, statesArrayImpBurn, statesArrayNoThrust), axis=0)
dataset_label = np.concatenate((labelsChemical, labelsElectric, labelsImpBurn, labelsNoThrust), axis=0)

indices = np.random.permutation(dataset.shape[0])

dataset = dataset[indices]
dataset_label = dataset_label[indices]

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

total_samples = len(dataset)
train_end = int(train_ratio * total_samples)
val_end = int((train_ratio + val_ratio) * total_samples)

# Split the data
train_data = dataset[:train_end]
train_label = dataset_label[:train_end]

val_data = dataset[train_end:val_end]
val_label = dataset_label[train_end:val_end]

test_data = dataset[val_end:]
test_label = dataset_label[val_end:]

train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label).squeeze(1).long())
val_dataset = TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_label).squeeze(1).long())
test_dataset = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label).squeeze(1).long())

train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True,pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False,pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False,pin_memory=True)


criterion = torch.nn.CrossEntropyLoss()
config = MambaConfig(d_model=input_size,n_layers = num_layers,expand_factor=hidden_size//input_size,d_state=32,d_conv=16,classifer=True)
model_mamba = MambaClassifier(config,input_size, hidden_size, num_layers, num_classes).to(device).double()
optimizer_mamba = torch.optim.Adam(model_mamba.parameters(), lr=learning_rate)

schedulerPatience = 5

scheduler_mamba = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_mamba,
    mode='min',             # or 'max' for accuracy
    factor=0.5,             # shrink LR by 50%
    patience=schedulerPatience             # wait for 3 epochs of no improvement
)


classlabels = ['No Thrust','Chemical','Electric','Impulsive']

if use_lstm:
    model_LSTM = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device).double()
    optimizer_LSTM = torch.optim.Adam(model_LSTM.parameters(), lr=learning_rate)
    scheduler_LSTM = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_LSTM,
        mode='min',             # or 'max' for accuracy
        factor=0.5,             # shrink LR by 50%
        patience=schedulerPatience
    )

    print('\nEntering LSTM Training Loop')
    LSTMTrainTime = timer()
    trainClassifier(model_LSTM,optimizer_LSTM,scheduler_LSTM,[train_loader,test_loader,val_loader],criterion,num_epochs,device)
    LSTMTrainTime.toc()
    printModelParmSize(model_LSTM)
    validateMultiClassClassifier(model_LSTM,val_loader,criterion,num_classes,device,classlabels)

print('\nEntering Mamba Training Loop')
mambaTrainTime = timer()
trainClassifier(model_mamba,optimizer_mamba,scheduler_mamba,[train_loader,test_loader,val_loader],criterion,num_epochs,device)
mambaTrainTime.toc()
printModelParmSize(model_mamba)
validateMultiClassClassifier(model_mamba,val_loader,criterion,num_classes,device,classlabels)
