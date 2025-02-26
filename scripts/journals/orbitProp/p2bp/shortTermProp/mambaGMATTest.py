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
from qutils.ml import trainModel, printModelParmSize, getDevice, Adam_mini, create_datasets, genPlotPrediction, LSTMSelfAttentionNetwork
from qutils.tictoc import timer
# from nets import Adam_mini

# from memory_profiler import profile
from qutils.mlExtras import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight

from qutils.mlSuperweight import findMambaSuperActivation, plotSuperActivation

compareLSTM = True
plotOn = False
printoutSuperweight = False
saveData = True

problemDim = 6

device = getDevice()

gmatImport = readGMATReport("gmat/data/reportLEO5050Prop.txt")
semimajorAxis = 7191.9388
tPeriod = 6083.4536377460845682560169095477
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
p_motion_knowledge = 5/4 * 3/4


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

timeToTrain = trainModel(model,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutAcc = True,printOutToc = True)

networkPrediction, testTime = plotStatePredictions(model,t,output_seq,train_in,test_in,train_size,test_size,DU=DU,TU=TU,outputToc=True)
output_seq = nonDim2Dim6(output_seq,DU,TU)

# networkPrediction = nonDim2Dim6(networkPrediction,DU,TU)

plotOrbitPhasePredictions(output_seq,networkPrediction)
plotOrbitPhasePredictions(output_seq,networkPrediction,plane='xz')
plotOrbitPhasePredictions(output_seq,networkPrediction,plane='yz')


plot3dOrbitPredictions(output_seq,networkPrediction)

print('total prop time',gmatImport[-1,-1])

# networkPrediction = nonDim2Dim6(networkPrediction,DU,TU)
# output_seq = nonDim2Dim6(output_seq,DU,TU)

# plotOrbitPredictions(output_seq,networkPrediction,t=t)
plotSolutionErrors(output_seq,networkPrediction,t/tPeriod)
plotPercentSolutionErrors(output_seq,networkPrediction,t/tPeriod,semimajorAxis,max(np.linalg.norm(gmatImport[:,3:6],axis=1)))
plotEnergy(output_seq,networkPrediction,t/tPeriod,orbitalEnergy,xLabel='Number of Periods (T)',yLabel='Specific Energy')
# plotDecAccs(decAcc,t,problemDim)

from qutils.mlExtras import rmse

rmseMamba = rmse(output_seq,networkPrediction)



errorAvg = np.nanmean(abs(networkPrediction-output_seq), axis=0)
print("Average values of each dimension:")
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg}")

printModelParmSize(model)
torchinfo.summary(model)


# round trip closure

finalConditions = networkPrediction[-1,:]
print(finalConditions)

if printoutSuperweight:
    printoutMaxLayerWeight(model)
    getSuperWeight(model)
    plotSuperWeight(model)

    magnitude, index = findMambaSuperActivation(model,test_in)
    plotSuperActivation(magnitude, index)


if compareLSTM:
    del model
    del optimizer
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    modelLSTM = returnModel('lstm')

    # optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    optimizer = Adam_mini(modelLSTM,lr=lr)

    criterion = F.smooth_l1_loss
    # criterion = torch.nn.HuberLoss()
    timeToTrainLSTM = trainModel(modelLSTM,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutAcc = True,printOutToc = True)

    output_seq = dim2NonDim6(output_seq,DU,TU)

    networkPredictionLSTM, testTimeLSTM = plotStatePredictions(modelLSTM,t,output_seq,train_in,test_in,train_size,test_size,DU=DU,TU=TU,outputToc=True)
    output_seq = nonDim2Dim6(output_seq,DU,TU)

    plot3dOrbitPredictions(output_seq,networkPrediction,earth=False,networkLabel="Mamba")
    plt.plot(networkPredictionLSTM[:, 0], networkPredictionLSTM[:, 1], networkPredictionLSTM[:, 2], label='LSTM',linestyle='dashed')
    plt.plot(0,0,0,"ko",label="Earth")
    plt.legend(fontsize=10)
    plt.tight_layout()

    plotOrbitPhasePredictions(output_seq,networkPredictionLSTM)
    plotOrbitPhasePredictions(output_seq,networkPredictionLSTM,plane='xz')
    plotOrbitPhasePredictions(output_seq,networkPredictionLSTM,plane='yz')

    plotSolutionErrors(output_seq,networkPredictionLSTM,t/tPeriod)

    fig, axes = newPlotSolutionErrors(output_seq,networkPrediction,t/tPeriod,timeLabel="Periods")
    newPlotSolutionErrors(output_seq,networkPredictionLSTM,t/tPeriod,timeLabel="Periods",newPlot=axes,networkLabels=["Mamba","LSTM"])
    mambaLine = mlines.Line2D([], [], color='b', label='Mamba')
    LSTMLine = mlines.Line2D([], [], color='orange', label='LSTM')
    fig.legend(handles=[mambaLine,LSTMLine])

    # plotPercentSolutionErrors(output_seq,networkPredictionLSTM,t/tPeriod,semimajorAxis,max(np.linalg.norm(gmatImport[:,3:6],axis=1)))

    plotEnergy(output_seq,networkPrediction,t/tPeriod,orbitalEnergy,xLabel='Number of Periods (T)',yLabel='Specific Energy')
    plt.plot(t/tPeriod,orbitalEnergy(networkPredictionLSTM),label='LSTM',linestyle='dashed')
    plt.legend()

    rmseLSTM = rmse(output_seq,networkPredictionLSTM)


    errorAvg = np.nanmean(abs(networkPredictionLSTM-output_seq), axis=0)
    print("Average values of each dimension:")
    for i, avg in enumerate(errorAvg, 1):
        print(f"Dimension {i}: {avg}")

    printModelParmSize(modelLSTM)
    torchinfo.summary(modelLSTM)

    finalConditions = networkPredictionLSTM[-1,:]
    print(finalConditions)

if plotOn is True:
    plt.show()

if saveData is True:
    import csv
    import os

    fieldnames = ["x","y","z","vx","vy","vz"]
    new_data_mamba = {"x":rmseMamba[0],"y":rmseMamba[1],"z":rmseMamba[2],"vx":rmseMamba[3],"vy":rmseMamba[4],"vz":rmseMamba[5]}
    new_data_LSTM = {"x":rmseLSTM[0],"y":rmseLSTM[1],"z":rmseLSTM[2],"vx":rmseLSTM[3],"vy":rmseLSTM[4],"vz":rmseLSTM[5]}


    file_path = 'p2bpShortRMSEMamba.csv'
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file,fieldnames=fieldnames)
        if not file_exists or os.path.getsize(file_path) == 0:
            writer.writeheader()
        writer.writerow(new_data_mamba)

    file_path = 'p2bpShortRMSELSTM.csv'
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file,fieldnames=fieldnames)
        if not file_exists or os.path.getsize(file_path) == 0:
            writer.writeheader()
        writer.writerow(new_data_LSTM)


    fieldnames = ["Mamba Train","LSTM Train","Mamba Test","LSTM Test"]
    new_data = {"Mamba Train":timeToTrain,"LSTM Train":timeToTrainLSTM,"Mamba Test":testTime,"LSTM Test":testTimeLSTM}


    file_path = 'p2bpShortTime.csv'
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file,fieldnames=fieldnames)
        if not file_exists or os.path.getsize(file_path) == 0:
            writer.writeheader()
        writer.writerow(new_data)