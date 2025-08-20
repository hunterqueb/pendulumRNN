import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
from sklearn.metrics import log_loss, classification_report, confusion_matrix

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
from qutils.ml.utils import getDevice, printModelParmSize
from qutils.ml.classifer import trainClassifier, LSTMClassifier, validateMultiClassClassifier
from qutils.ml.mamba import Mamba, MambaConfig, MambaClassifier
from qutils.ml.superweight import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight, findMambaSuperActivation,plotSuperActivation
from qutils.orbital import dim2NonDim6

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no-lstm',dest="use_lstm", action='store_false', help='Use LSTM model')
parser.add_argument("--systems", type=int, default=10000, help="Number of random systems to access")
parser.add_argument("--propMin", type=int, default=30, help="Minimum propagation time in minutes")
parser.add_argument("--orbit", type=str, default="vleo", help="Orbit type: vleo, leo")
parser.add_argument("--test", type=str, default=None, help="Orbit type for test set: vleo, leo")
parser.add_argument("--OE", action='store_true', help="Use OE elements instead of ECI states")
parser.add_argument("--noise", action='store_true', help="Add noise to the data")
parser.add_argument("--norm", action='store_true', help="Normalize the semi-major axis by Earth's radius")
parser.add_argument("--one-shot",type=str, default=None, help="Use one shot transfer learning. Takes in a path to a saved pt model")
parser.add_argument("--one-pass",dest="one_pass",action='store_true', help="Use one pass learning.")
parser.add_argument("--save",dest="save_to_log",action="store_true",help="output console printout to log file in the same location as datasets")
parser.add_argument("--energy",dest="use_energy",action="store_true",help="Use energy as a feature.")
parser.add_argument("--hybrid",dest="use_hybrid",action="store_true",help="Use a hybrid network.")
parser.add_argument("--superweight",dest="find_SW",action="store_true",help="Superweight analysis")
parser.add_argument("--classic",dest="use_classic",action="store_true",help="Use classic ML classification for comparison")
parser.add_argument("--nearest",dest="use_nearestNeighbor",action="store_true",help="Use classic ML classification (1-nearest neighbor w/ DTW) for comparison")

parser.set_defaults(use_lstm=True)
parser.set_defaults(OE=False)
parser.set_defaults(noise=False)
parser.set_defaults(norm=False)
parser.set_defaults(one_pass=False)
parser.set_defaults(save_to_log=False)
parser.set_defaults(use_energy=False)
parser.set_defaults(use_hybrid=False)
parser.set_defaults(find_SW=False)
parser.set_defaults(use_classic=False)
parser.set_defaults(use_nearestNeighbor=False)

args = parser.parse_args()
use_lstm = args.use_lstm
numMinProp = args.propMin
numRandSys = args.systems
orbitType = args.orbit
if args.test is None:
    args.test = args.orbit
testSet = args.test
useOE = args.OE
useNoise = args.noise
useNorm = args.norm
useOneShot = args.one_shot
useOnePass = args.one_pass
save_to_log = args.save_to_log
useEnergy=args.use_energy
useHybrid=args.use_hybrid
find_SW=args.find_SW
use_classic = args.use_classic
use_nearestNeighbor = args.use_nearestNeighbor

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
    if use_classic:
        strAdd = strAdd + "DT"
    if use_nearestNeighbor:
        strAdd = strAdd + "1-NN"
    if testSet != orbitType:
        strAdd = strAdd + "Test" + testSet
    logFileLoc = dataLoc+"/"+str(numMinProp) + "min" + str(numRandSys)+ strAdd +'.log'
    print("saving log output to {}".format(logFileLoc))

    # file to open
    f = open(logFileLoc, 'w')
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
hidden_factor = 8  # hidden size is a multiple of input size
hidden_size = int(input_size * hidden_factor) # must be multiple of train dim
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
    hidden_size = int(input_size * hidden_factor) 
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

if testSet != orbitType:
    # if using a different orbit type for the test set, load the test set from the other orbit type
    dataLoc = "gmat/data/classification/"+ testSet +"/" + str(numMinProp) + "min-" + str(numRandSys)
    # load the test set from the other orbit type
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
    if useEnergy:
        dataset_test = np.concatenate((energyChemical, energyElectric, energyImpBurn, energyNoThrust), axis=0)
    if useEnergy and useOE:
        combinedChemical = np.concatenate((statesArrayChemical,energyChemical),axis=2) 
        combinedElectric = np.concatenate((statesArrayElectric,energyElectric),axis=2) 
        combinedImpBurn = np.concatenate((statesArrayImpBurn,energyImpBurn),axis=2) 
        combinedNoThrust = np.concatenate((statesArrayNoThrust,energyNoThrust),axis=2) 
        dataset_test = np.concatenate((combinedChemical, combinedElectric, combinedImpBurn, combinedNoThrust), axis=0)
        input_size = 6 + 1
        config = MambaConfig(d_model=input_size,n_layers = num_layers,expand_factor=hidden_size//input_size,d_state=32,d_conv=16,classifer=True)

    dataset_label_test = np.concatenate((labelsChemical, labelsElectric, labelsImpBurn, labelsNoThrust), axis=0)

    dataset_test = dataset[indices]
    dataset_label_test = dataset_label[indices]

    test_data = dataset_test[val_end:]
    test_label = dataset_label_test[val_end:]

else:
    test_data = dataset[val_end:]
    test_label = dataset_label[val_end:]

train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label).squeeze(1).long())
val_dataset = TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_label).squeeze(1).long())
test_dataset = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label).squeeze(1).long())

train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True,pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False,pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False,pin_memory=True)

class CostSensitiveCELoss(nn.Module):
    def __init__(self, cost_matrix: torch.Tensor):
        super().__init__()
        self.register_buffer("cost_matrix", cost_matrix)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Softmax probabilities
        probs = F.softmax(logits, dim=1)  # shape: [batch_size, num_classes]

        # Gather cost vectors for each target in the batch
        cost_vectors = self.cost_matrix[targets]  # shape: [batch_size, num_classes]

        # Compute the expected cost per sample
        expected_cost = torch.sum(probs * cost_vectors, dim=1)  # shape: [batch_size]

        # Return mean cost
        return expected_cost.mean()

class BlendedLoss(nn.Module):
    def __init__(self, alpha: float, cost_matrix: torch.Tensor):
        super().__init__()
        assert 0.0 <= alpha <= 1.0
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.cost_ce = CostSensitiveCELoss(cost_matrix)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        cs_loss = self.cost_ce(logits, targets)
        return self.alpha * ce_loss + (1 - self.alpha) * cs_loss

# cost[i][j] = penalty for predicting j when true class is i

# cost_matrix = torch.tensor([
#     [0.0, 1.0, 0.2, 1.0],  # True class 0
#     [1.0, 0.0, 1.0, 1.0],  # True class 1
#     [0.2, 1.0, 0.0, 1.0],  # True class 2
#     [1.0, 1.0, 1.0, 0.0],  # True class 3
# ], dtype=torch.float32)


cost_matrix = torch.tensor([
    [0.0, 0.9, 0.1, 0.9],
    [0.9, 0.0, 0.9, 0.9],
    [0.9, 0.9, 0.0, 0.9],
    [0.9, 0.2, 0.9, 0.0]], dtype=torch.float32).to(device)

cost_matrix = cost_matrix / cost_matrix.max()

alpha = 0
criterion = BlendedLoss(alpha=alpha, cost_matrix=cost_matrix)
# criterion = torch.nn.CrossEntropyLoss()


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

    if testSet != orbitType:
        validateMultiClassClassifier(model_hybrid,test_loader,criterion,num_classes,device,classlabels,printReport=True)
    else:
        validateMultiClassClassifier(model_hybrid,val_loader,criterion,num_classes,device,classlabels,printReport=True)

if use_classic:
    from lightgbm import LGBMClassifier

    print("\nEntering Decision Trees Training Loop")
    DTTimer = timer()
    def printClassicModelSize(model):
        import tempfile, pathlib

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "model.bin"   # any extension is fine
            model.booster_.save_model(str(path))     # binary dump by default
            size_bytes = path.stat().st_size
        print("\n==========================================================================================")
        print(f"Total parameters: NaN")
        print(f"Total memory (bytes): {size_bytes}")
        print(f"Total memory (MB): {size_bytes / (1024 ** 2)}")
        print("==========================================================================================")

    def validate_lightgbm(model, val_loader, num_classes, classlabels=None, print_report=True):
        """Evaluate a trained LightGBM multiclass classifier on a PyTorch‑style DataLoader.

        * model          - fitted lightgbm.LGBMClassifier (objective='multiclass')
        * val_loader     - yields (seq_batch, label_batch); seq_batch can be torch.Tensor or np.ndarray
                        Shape per sample must match training: (7, L).  Flatten before predict.
        * num_classes    - integer (4 in your case)
        """
        # --------------------------------------------------------------------- #
        # Aggregate validation data                                             #
        # --------------------------------------------------------------------- #
        X_list, y_list = [], []
        for seq, lab in val_loader:
            # → ndarray, shape (batch, 7*L)
            xb = (seq if isinstance(seq, np.ndarray) else seq.cpu().numpy()).reshape(seq.shape[0], -1)
            yb = (lab if isinstance(lab, np.ndarray) else lab.cpu().numpy())
            X_list.append(xb)
            y_list.append(yb)

        X_val = np.concatenate(X_list, axis=0)
        y_true = np.concatenate(y_list, axis=0)

        # --------------------------------------------------------------------- #
        # Predict                                                               #
        # --------------------------------------------------------------------- #
        proba = model.predict_proba(X_val, num_iteration=model.best_iteration_)
        y_pred = proba.argmax(axis=1)

        # --------------------------------------------------------------------- #
        # Metrics                                                               #
        # --------------------------------------------------------------------- #
        val_loss = log_loss(y_true, proba, labels=np.arange(num_classes))
        accuracy = 100.0 * (y_pred == y_true).mean()

        # Per‑class accuracy
        class_tot = np.bincount(y_true, minlength=num_classes)
        class_corr = np.bincount(y_true[y_true == y_pred], minlength=num_classes)
        per_class_acc = 100.0 * class_corr / np.maximum(class_tot, 1)

        # --------------------------------------------------------------------- #
        # Reporting                                                             #
        # --------------------------------------------------------------------- #
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%\n")

        print("Per-Class Validation Accuracy:")
        for i in range(num_classes):
            label = classlabels[i] if classlabels else f"Class {i}"
            if class_tot[i]:
                print(f"  {label}: {per_class_acc[i]:.2f}% ({class_corr[i]}/{class_tot[i]})")
            else:
                print(f"  {label}: No samples")

        if print_report:
            print("\nClassification Report:")
            print(
                classification_report(
                    y_true, y_pred,
                    labels=list(range(num_classes)),
                    target_names=(classlabels if classlabels else None),
                    digits=4,
                    zero_division=0,
                )
            )

            cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
            print("\nConfusion Matrix (rows = true, cols = predicted):")
            print(
                pd.DataFrame(
                    cm,
                    index=[f"T_{cls}" for cls in (classlabels if classlabels else range(num_classes))],
                    columns=[f"P_{cls}" for cls in (classlabels if classlabels else range(num_classes))]
                )
            )

        return val_loss, accuracy
    classicModel = LGBMClassifier(objective="multiclass",num_classes=num_classes,n_estimators=100,max_depth=-1,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,verbosity=-1)   # or 'verbose' for older builds)
    
    # flatten features
    X_train = train_data.reshape(train_data.shape[0], -1).astype(np.float32)    # (number of systems to train on, network features * length of time series)    
    y_train = train_label.reshape(-1).astype(np.int32)             # (number of systems to train on,)
    classicModel.fit(X_train, y_train)
    DTTimer.toc()
    printClassicModelSize(classicModel)
    if testSet != orbitType:
        validate_lightgbm(classicModel, test_loader, num_classes, classlabels=classlabels, print_report=True)
    else:
        validate_lightgbm(classicModel, val_loader, num_classes, classlabels=classlabels, print_report=True)

if use_nearestNeighbor:
    def z_normalize(ts, eps=1e-8):
        # ts: [T] or [T,C]
        mean = ts.mean(axis=0, keepdims=True)
        std = ts.std(axis=0, keepdims=True)
        return (ts - mean) / (std + eps)

    def train_data_z_normalize(train_data):
        """Z-normalize training data along the time axis."""
        return np.array([z_normalize(ts) for ts in train_data])

    def print1_NNModelSize(model):
        import tempfile, pathlib
        import pickle

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "model.pkl"
            with open(path, "wb") as f:
                pickle.dump(model, f)
            size_bytes = path.stat().st_size

        print("\n" + "=" * 90)
        print(f"Total parameters: NaN (non-parametric model)")
        print(f"Total memory (bytes): {size_bytes}")
        print(f"Total memory (MB): {size_bytes / (1024 ** 2):.4f}")
        print("=" * 90)

    def validate_1NN(clf, val_loader, num_classes, classlabels=None):
        """Evaluate a 1-NN classifier (e.g., sktime KNeighborsTimeSeriesClassifier) on a PyTorch DataLoader."""
        X_val_list, y_val_list = [], []

        for seq, lab in val_loader:
            xb = seq.cpu().numpy()  # preserve time-series shape
            yb = lab.cpu().numpy()
            X_val_list.append(xb) #z-normalize each time series
            y_val_list.append(yb)

        # Merge batches
        X_val_np = np.concatenate(X_val_list, axis=0)
        y_true = np.concatenate(y_val_list)

        # Adapt shape for sktime: [N,C,T]
        # [N,T,C] → [N,C,T]
        X_val_np = np.transpose(X_val_np, (0, 2, 1))

        # Predict
        y_pred = clf.predict(X_val_np)

        # Accuracy
        correct = (y_pred == y_true).sum()
        total = len(y_true)
        accuracy = 100.0 * correct / total

        print(f"Validation Loss: NaN, Validation Accuracy: {accuracy:.2f}%\n")

        # Per-class accuracy
        class_corr = np.zeros(num_classes, dtype=int)
        class_tot = np.zeros(num_classes, dtype=int)
        for yt, yp in zip(y_true, y_pred):
            class_tot[yt] += 1
            if yt == yp:
                class_corr[yt] += 1
        per_class_acc = 100.0 * class_corr / np.maximum(class_tot, 1)

        print("Per-Class Validation Accuracy:")
        for i in range(num_classes):
            label = classlabels[i] if classlabels else f"Class {i}"
            if class_tot[i]:
                print(f"  {label}: {per_class_acc[i]:.2f}% ({class_corr[i]}/{class_tot[i]})")
            else:
                print(f"  {label}: No samples")

        print("\nClassification Report:")
        print(
            classification_report(
                y_true, y_pred,
                labels=list(range(num_classes)),
                target_names=(classlabels if classlabels else None),
                digits=4,
                zero_division=0,
            )
        )

        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        print("\nConfusion Matrix (rows = true, cols = predicted):")
        print(
            pd.DataFrame(
                cm,
                index=[f"T_{cls}" for cls in (classlabels if classlabels else range(num_classes))],
                columns=[f"P_{cls}" for cls in (classlabels if classlabels else range(num_classes))]
            )
        )

    from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

    print("\nEntering Nearest Neighbor Training Loop")
    dtw = timer()
    # [N,T,C] -> [N,C,T]
    train_data_NN = np.transpose(train_data, (0, 2, 1))

    # train_data_NN = train_data_z_normalize(train_data_NN)  # Z-normalize along time axis

    clf = KNeighborsTimeSeriesClassifier(
        n_neighbors=1,
        distance="dtw",
        distance_params={"sakoe_chiba_radius": 10}
 )
    clf.fit(train_data_NN, train_label)
    dtw.toc()
    print1_NNModelSize(clf)
    if testSet != orbitType:
        validate_1NN(clf, test_loader, num_classes, classlabels=classlabels)
    else:
        validate_1NN(clf, val_loader, num_classes, classlabels=classlabels)

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
    validateMultiClassClassifier(model_LSTM,val_loader,criterion,num_classes,device,classlabels,printReport=True)
    if testSet != orbitType:
        validateMultiClassClassifier(model_LSTM,test_loader,criterion,num_classes,device,classlabels,printReport=True)
    else:
        validateMultiClassClassifier(model_LSTM,val_loader,criterion,num_classes,device,classlabels,printReport=True)

print('\nEntering Mamba Training Loop')
trainClassifier(model_mamba,optimizer_mamba,scheduler_mamba,[train_loader,test_loader,val_loader],criterion,num_epochs,device)
printModelParmSize(model_mamba)
if testSet != orbitType:
    validateMultiClassClassifier(model_mamba,test_loader,criterion,num_classes,device,classlabels,printReport=True)
else:
    validateMultiClassClassifier(model_mamba,val_loader,criterion,num_classes,device,classlabels,printReport=True)
# torch.save(model_mamba.state_dict(), f"{dataLoc}/mambaTimeSeriesClassificationGMATThrusts"+ orbitType +".pt")

if find_SW:
    magnitude, index = findMambaSuperActivation(model_mamba,torch.tensor(test_data).to(device))
    # super activation returns the entire mamba network parameters, but the classifier does not use the out_proj layer
    # so we drop it
    magnitude = magnitude[:-1]
    index = index[:-1]
    # also drop the x_proj layer, no longer needed as well
    magnitude.pop(2)
    index.pop(2)


    normedMagsMRP = np.zeros((len(magnitude),))
    for i in range(len(magnitude)):
        normedMagsMRP[i] = magnitude[i].norm().detach().cpu()

    printoutMaxLayerWeight(model_mamba)
    getSuperWeight(model_mamba)
    plotSuperWeight(model_mamba)
    plotSuperActivation(magnitude, index,printOutValues=True,mambaLayerAttributes = ["in_proj","conv1d","dt_proj"])
    plt.title("Mamba Classifier Super Activations")


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


plt.show()