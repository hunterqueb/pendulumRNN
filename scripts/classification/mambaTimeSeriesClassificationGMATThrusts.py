import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn


# script usage

# call the script from the main folder directory, adding --save saves the output to a log file in the location of the datasets
# $ python scripts/classification/mambaTimeSeriesClassificationGMATThrusts.py \
# --systems 10000 --propMin 5 --OE --norm --orbit vleo 

# display the data by calling the displayLogData.py script from its contained folder

class HybridClassifier(nn.Module):
    def __init__(self,config, input_size, hidden_size, num_layers, num_classes):
        super(HybridClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # Bidirectional LSTM
        )
        self.mamba = Mamba(config)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        """
        x: [batch_size, seq_length, input_size]
        """
        # h0, c0 default to zero if not provided
        out, (h_n, c_n) = self.lstm(x)
        h_n = self.mamba(out) # [batch_size, seq_length, hidden_size]

        # h_n is shape [num_layers, batch_size, hidden_size].
        # We typically take the last layer's hidden state: h_n[-1]
        last_hidden = h_n[:,-1,:]  # [batch_size, hidden_size]
        
        # Pass the last hidden state through a linear layer for classification
        logits = self.fc(last_hidden)  # [batch_size, num_classes]
        
        return logits


from qutils.tictoc import timer
from qutils.ml import getDevice, trainClassifier, LSTMClassifier, MambaClassifier, printModelParmSize, validateMultiClassClassifier
from qutils.mamba import Mamba, MambaConfig
from qutils.mlExtras import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight
from qutils.mlSuperweight import findMambaSuperActivation,plotSuperActivation,zeroModelWeight
from qutils.orbital import dim2NonDim6

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no-lstm',dest="use_lstm", action='store_false', help='Use LSTM model')
parser.add_argument("--systems", type=int, default=10000, help="Number of random systems to access")
parser.add_argument("--propMin", type=int, default=30, help="Minimum propagation time in minutes")
parser.add_argument("--orbit", type=str, default="vleo", help="Orbit type: vleo, leo")
parser.add_argument("--OE", action='store_true', help="Use OE elements instead of ECI states")
parser.add_argument("--noise", action='store_true', help="Add noise to the data")
parser.add_argument("--norm", action='store_true', help="Normalize the semi-major axis by Earth's radius")
parser.add_argument("--one-shot",type=str, default=None, help="Use one shot transfer learning. Takes in a path to a saved pt model")
parser.add_argument("--one-pass",dest="one_pass",action='store_true', help="Use one pass learning.")
parser.add_argument("--save",dest="save_to_log",action="store_true",help="output console printout to log file in the same location as datasets")
parser.add_argument("--energy",dest="use_energy",action="store_true",help="Use energy as a feature.")
parser.add_argument("--hybrid",dest="use_hybrid",action="store_true",help="Use a hybrid network.")

parser.set_defaults(use_lstm=True)
parser.set_defaults(OE=False)
parser.set_defaults(noise=False)
parser.set_defaults(norm=False)
parser.set_defaults(one_pass=False)
parser.set_defaults(save_to_log=False)
parser.set_defaults(use_energy=False)
parser.set_defaults(use_hybrid=False)

args = parser.parse_args()
use_lstm = args.use_lstm
numMinProp = args.propMin
numRandSys = args.systems
orbitType = args.orbit
useOE = args.OE
useNoise = args.noise
useNorm = args.norm
useOneShot = args.one_shot
useOnePass = args.one_pass
save_to_log = args.save_to_log
useEnergy=args.use_energy
useHybrid=args.use_hybrid

dataLoc = "gmat/data/classification/"+ orbitType +"/" + str(numMinProp) + "min-" + str(numRandSys)

if save_to_log:
    import sys

    strAdd = ""
    if useEnergy:
        strAdd = "Energy"
    if useOE:
        strAdd = strAdd + "OE"
    if useNorm:
        strAdd = strAdd + "Norm"
    if useNoise:
        strAdd = strAdd + "Noise"
    if useOneShot:
        strAdd = strAdd + "OneShot"
    if useOnePass:
        strAdd = strAdd + "OnePass"
    if useHybrid:
        strAdd = strAdd + "Hybrid"

    logFileLoc = dataLoc+"/"+str(numMinProp) + "min" + str(numRandSys)+ strAdd +'.log'
    print("saving log output to {}".format(logFileLoc))

    # file to open
    f = open(logFileLoc, 'a')
    # change stdout to write to file -- this allows for printing from functions to a file
    sys.stdout = f

def apply_noise(data, pos_noise_std, vel_noise_std):
    mid = data.shape[1] // 2  # Split index
    pos_noise = np.random.normal(0, pos_noise_std, size=data[:, :mid].shape)
    vel_noise = np.random.normal(0, vel_noise_std, size=data[:, mid:].shape)
    noisy_data = data.copy()
    noisy_data[:, :mid] += pos_noise
    noisy_data[:, mid:] += vel_noise
    return noisy_data


# get npz files in folder and load them into script
if useOE:
    a = np.load(f"{dataLoc}/OEArrayChemical.npz")
    statesArrayChemical = a['OEArrayChemical'][:,:,0:6]
    a = np.load(f"{dataLoc}/OEArrayElectric.npz")
    statesArrayElectric = a['OEArrayElectric'][:,:,0:6]
    a = np.load(f"{dataLoc}/OEArrayImpBurn.npz")
    statesArrayImpBurn = a['OEArrayImpBurn'][:,:,0:6]
    a = np.load(f"{dataLoc}/OEArrayNoThrust.npz")
    statesArrayNoThrust = a['OEArrayNoThrust'][:,:,0:6]

    if useNoise:
        statesArrayChemical = apply_noise(statesArrayChemical, 1e-3, 1e-3)
        statesArrayElectric = apply_noise(statesArrayElectric, 1e-3, 1e-3)
        statesArrayImpBurn = apply_noise(statesArrayImpBurn, 1e-3, 1e-3)
        statesArrayNoThrust = apply_noise(statesArrayNoThrust, 1e-3, 1e-3)
    if useNorm:
        R = 6378.1363 # km
        statesArrayChemical[:,:,0] = statesArrayChemical[:,:,0] / R
        statesArrayElectric[:,:,0] = statesArrayElectric[:,:,0] / R
        statesArrayImpBurn[:,:,0] = statesArrayImpBurn[:,:,0] / R
        statesArrayNoThrust[:,:,0] = statesArrayNoThrust[:,:,0] / R

else:
    a = np.load(f"{dataLoc}/statesArrayChemical.npz")
    statesArrayChemical = a['statesArrayChemical']

    a = np.load(f"{dataLoc}/statesArrayElectric.npz")
    statesArrayElectric = a['statesArrayElectric']

    a = np.load(f"{dataLoc}/statesArrayImpBurn.npz")
    statesArrayImpBurn = a['statesArrayImpBurn']

    a = np.load(f"{dataLoc}/statesArrayNoThrust.npz")
    statesArrayNoThrust = a['statesArrayNoThrust']

    if useNoise:
        statesArrayChemical = apply_noise(statesArrayChemical, 1e-3, 1e-3)
        statesArrayElectric = apply_noise(statesArrayElectric, 1e-3, 1e-3)
        statesArrayImpBurn = apply_noise(statesArrayImpBurn, 1e-3, 1e-3)
        statesArrayNoThrust = apply_noise(statesArrayNoThrust, 1e-3, 1e-3)
    if useNorm:
        for i in range(statesArrayChemical.shape[0]):
            statesArrayChemical[i,:,:] = dim2NonDim6(statesArrayChemical[i,:,:])
            statesArrayElectric[i,:,:] = dim2NonDim6(statesArrayElectric[i,:,:])
            statesArrayImpBurn[i,:,:] = dim2NonDim6(statesArrayImpBurn[i,:,:])
            statesArrayNoThrust[i,:,:] = dim2NonDim6(statesArrayNoThrust[i,:,:])
del a

device = getDevice()

batchSize = 16
problemDim = 6

# Hyperparameters
input_size = problemDim 
hidden_size = 48 # must be multiple of train dim
num_layers = 1
num_classes = 4  # e.g., multiclass classification
learning_rate = 1e-3
num_epochs = 100

if useOnePass:
    num_epochs = 1

if useEnergy:
    from qutils.orbital import orbitalEnergy
    problemDim = 1
    input_size = 1

    energyChemical = np.zeros((statesArrayChemical.shape[0],statesArrayChemical.shape[1],1))
    energyElectric= np.zeros((statesArrayChemical.shape[0],statesArrayChemical.shape[1],1))
    energyImpBurn= np.zeros((statesArrayChemical.shape[0],statesArrayChemical.shape[1],1))
    energyNoThrust= np.zeros((statesArrayChemical.shape[0],statesArrayChemical.shape[1],1))
    for i in range(statesArrayChemical.shape[0]):
        energyChemical[i,:,0] = orbitalEnergy(statesArrayChemical[i,:,:])
        energyElectric[i,:,0] = orbitalEnergy(statesArrayElectric[i,:,:])
        energyImpBurn[i,:,0] = orbitalEnergy(statesArrayImpBurn[i,:,:])
        energyNoThrust[i,:,0] = orbitalEnergy(statesArrayNoThrust[i,:,:])
    if useNorm:
        normingEnergy = energyNoThrust[0,0,0]
        energyChemical[:,:,0] = energyChemical[:,:,0] / normingEnergy
        energyElectric[:,:,0] = energyElectric[:,:,0] / normingEnergy
        energyImpBurn[:,:,0] = energyImpBurn[:,:,0] / normingEnergy
        energyNoThrust[:,:,0] = energyNoThrust[:,:,0] / normingEnergy
    # plt.figure()
    # plt.plot(energyChemical[0,:,:],label="Chemical")
    # plt.plot(energyElectric[0,:,:],label="Electric")
    # plt.plot(energyImpBurn[0,:,:],label="Impulsive")
    # plt.plot(energyNoThrust[0,:,:],label="No Thrust")
    # plt.grid()
    # plt.title("Normalized Energy By {:4f}".format(normingEnergy))
    # plt.legend()
    # plt.show()


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

if useEnergy:
    dataset = np.concatenate((energyChemical, energyElectric, energyImpBurn, energyNoThrust), axis=0)
if useEnergy and useOE:
    combinedChemical = np.concatenate((statesArrayChemical,energyChemical),axis=2) 
    combinedElectric = np.concatenate((statesArrayElectric,energyElectric),axis=2) 
    combinedImpBurn = np.concatenate((statesArrayImpBurn,energyImpBurn),axis=2) 
    combinedNoThrust = np.concatenate((statesArrayNoThrust,energyNoThrust),axis=2) 
    dataset = np.concatenate((combinedChemical, combinedElectric, combinedImpBurn, combinedNoThrust), axis=0)
    input_size = 6 + 1
    hidden_size = int(input_size * 8) 
    config = MambaConfig(d_model=input_size,n_layers = num_layers,expand_factor=hidden_size//input_size,d_state=32,d_conv=16,classifer=True)

dataset_label = np.concatenate((labelsChemical, labelsElectric, labelsImpBurn, labelsNoThrust), axis=0)

indices = np.random.permutation(dataset.shape[0])

dataset = dataset[indices]
dataset_label = dataset_label[indices]

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
model_mamba = MambaClassifier(config,input_size, hidden_size, num_layers, num_classes).to(device).double()

if useOneShot is not None:
    train_ratio = 0.1
    # split the rest into validation and test sets evenly 
    val_ratio = 0.5 * (1 - train_ratio)
    test_ratio = 0.5 * (1 - train_ratio)
    model_mamba.load_state_dict(torch.load(useOneShot, weights_only=True))
    print("Number of training samples in one-shot transfer learning:", int(train_ratio * dataset.shape[0]))
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
optimizer_mamba = torch.optim.Adam(model_mamba.parameters(), lr=learning_rate)

schedulerPatience = 5

scheduler_mamba = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_mamba,
    mode='min',             # or 'max' for accuracy
    factor=0.5,             # shrink LR by 50%
    patience=schedulerPatience             # wait for 3 epochs of no improvement
)


classlabels = ['No Thrust','Chemical','Electric','Impulsive']

if useHybrid:
    config_hybrid = MambaConfig(d_model=hidden_size * 2,n_layers = 1,expand_factor=1,d_state=32,d_conv=16,classifer=True)

    model_hybrid = HybridClassifier(config_hybrid,input_size,hidden_size,num_layers,num_classes).to(device).double()
    optimizer_hybrid = torch.optim.Adam(model_hybrid.parameters(), lr=learning_rate)
    scheduler_hybrid = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_hybrid,
        mode='min',             # or 'max' for accuracy
        factor=0.5,             # shrink LR by 50%
        patience=schedulerPatience
    )

    print('\nEntering Hybrid Training Loop')
    trainClassifier(model_hybrid,optimizer_hybrid,scheduler_hybrid,[train_loader,test_loader,val_loader],criterion,num_epochs,device)
    printModelParmSize(model_hybrid)
    validateMultiClassClassifier(model_hybrid,val_loader,criterion,num_classes,device,classlabels)


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
    trainClassifier(model_LSTM,optimizer_LSTM,scheduler_LSTM,[train_loader,test_loader,val_loader],criterion,num_epochs,device)
    printModelParmSize(model_LSTM)
    validateMultiClassClassifier(model_LSTM,val_loader,criterion,num_classes,device,classlabels)

print('\nEntering Mamba Training Loop')
trainClassifier(model_mamba,optimizer_mamba,scheduler_mamba,[train_loader,test_loader,val_loader],criterion,num_epochs,device)
printModelParmSize(model_mamba)
validateMultiClassClassifier(model_mamba,val_loader,criterion,num_classes,device,classlabels)
# torch.save(model_mamba.state_dict(), f"{dataLoc}/mambaTimeSeriesClassificationGMATThrusts"+ orbitType +".pt")




# # example onnx export
# # # generate example inputs for ONNX export
# example_inputs = torch.randn(1, numMinProp, input_size).to(device).double()
# # export the model to ONNX format
# # Note: `dynamo=True` is used to enable PyTorch's dynamo for better performance and compatibility.
# onnx_path = f"{dataLoc}/mambaTimeSeriesClassificationGMATThrusts.onnx"
# onnx_program = torch.onnx.export(model_mamba, example_inputs,onnx_path)
# print(f"ONNX model saved to {onnx_path}")

if save_to_log:
    sys.stdout = sys.__stdout__    # or your saved original_stdout
    f.close()