import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo

from qutils.integrators import ode85
from qutils.plot import plot3dOrbitPredictions,plotOrbitPhasePredictions, plotSolutionErrors,plotPercentSolutionErrors, plotEnergy,plotStatePredictions,newPlotSolutionErrors
from qutils.mlExtras import findDecAcc
from qutils.orbital import nonDim2Dim6, returnCR3BPIC, readGMATReport, dim2NonDim6, orbitalEnergy
from qutils.mamba import Mamba, MambaConfig
from qutils.ml import printModelParmSize, getDevice, Adam_mini, create_datasets, genPlotPrediction, LSTMSelfAttentionNetwork
from qutils.tictoc import timer
# from nets import Adam_mini

# from memory_profiler import profile
from qutils.mlExtras import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight

compareLSTM = True
plotOn = True
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
train_in,train_out,test_in,test_out = create_datasets(output_seq,1,train_size,device)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

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

trainTime = timer()
for epoch in range(n_epochs):

    # trajPredition = plotPredition(epoch,model,'target',t=t*TU,output_seq=pertNR)

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


networkPrediction = plotStatePredictions(model,t,output_seq,train_in,test_in,train_size,test_size,DU=DU,TU=TU)
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
errorAvg = np.nanmean(abs(networkPrediction-output_seq), axis=0)
print("Average values of each dimension:")
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg}")

printModelParmSize(model)
torchinfo.summary(model)

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
    trainTime = timer()
    for epoch in range(n_epochs):

        # trajPredition = plotPredition(epoch,model,'target',t=t*TU,output_seq=pertNR)

        modelLSTM.train()
        for X_batch, y_batch in loader:
            y_pred = modelLSTM(X_batch)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        modelLSTM.eval()
        with torch.no_grad():
            y_pred_train = modelLSTM(train_in)
            train_loss = np.sqrt(criterion(y_pred_train, train_out).cpu())
            y_pred_test = modelLSTM(test_in)
            test_loss = np.sqrt(criterion(y_pred_test, test_out).cpu())

            decAcc, err1 = findDecAcc(train_out,y_pred_train,printOut=False)
            decAcc, err2 = findDecAcc(test_out,y_pred_test)
            err = np.concatenate((err1,err2),axis=0)

        print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))
    trainTime.toc()


    output_seq = dim2NonDim6(output_seq,DU,TU)

    networkPredictionLSTM = plotStatePredictions(modelLSTM,t,output_seq,train_in,test_in,train_size,test_size,DU=DU,TU=TU)
    output_seq = nonDim2Dim6(output_seq,DU,TU)

    plot3dOrbitPredictions(output_seq,networkPrediction,earth=False,networkLabel="Mamba")
    plt.plot(networkPredictionLSTM[:, 0], networkPredictionLSTM[:, 1], networkPredictionLSTM[:, 2], label='LSTM')
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
    fig.tight_layout()

    # plotPercentSolutionErrors(output_seq,networkPredictionLSTM,t/tPeriod,semimajorAxis,max(np.linalg.norm(gmatImport[:,3:6],axis=1)))

    plotEnergy(output_seq,networkPrediction,t/tPeriod,orbitalEnergy,xLabel='Number of Periods (T)',yLabel='Specific Energy')
    plt.plot(t/tPeriod,orbitalEnergy(networkPredictionLSTM),label='LSTM')
    plt.legend()

    errorAvg = np.nanmean(abs(networkPredictionLSTM-output_seq), axis=0)
    print("Average values of each dimension:")
    for i, avg in enumerate(errorAvg, 1):
        print(f"Dimension {i}: {avg}")

    printModelParmSize(modelLSTM)
    torchinfo.summary(modelLSTM)

if printoutSuperweight is True:
    printoutMaxLayerWeight(model)
    getSuperWeight(model)
    plotSuperWeight(model)

if plotOn is True:
    plt.show()