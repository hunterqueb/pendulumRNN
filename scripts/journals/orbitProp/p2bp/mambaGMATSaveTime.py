
import torch
import torch.nn.functional as F
import os


from qutils.plot import plotStatePredictions
from qutils.orbital import readGMATReport, dim2NonDim6, nonDim2Dim6
from qutils.ml.mamba import Mamba, MambaConfig
from qutils.ml.utils import getDevice, Adam_mini, rmse
from qutils.ml.regression import trainModel, create_datasets, LSTMSelfAttentionNetwork
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

networkPrediction,testTime = plotStatePredictions(model,t,output_seq,train_in,test_in,train_size,test_size,DU=DU,TU=TU,plotOn=False,outputToc = True)
# output_seq = nonDim2Dim6(output_seq,DU,TU)

del model
del optimizer
torch.cuda.empty_cache()
import gc
gc.collect()
modelLSTM = returnModel('lstm')

optimizer = Adam_mini(modelLSTM,lr=lr)

timeToTrainLSTM = trainModel(modelLSTM,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutAcc = True,printOutToc = True)

# output_seq = dim2NonDim6(output_seq,DU,TU)

networkPredictionLSTM,testTimeLSTM = plotStatePredictions(modelLSTM,t,output_seq,train_in,test_in,train_size,test_size,DU=DU,TU=TU,plotOn=False,outputToc = True)
# output_seq = nonDim2Dim6(output_seq,DU,TU)
import csv

fieldnames = ["Mamba Train","LSTM Train","Mamba Test","LSTM Test"]
new_data = {"Mamba Train":timeToTrain,"LSTM Train":timeToTrainLSTM,"Mamba Test":testTime,"LSTM Test":testTimeLSTM}


file_path = 'p2bp.csv'
file_exists = os.path.isfile(file_path)

with open(file_path, 'a', newline='') as file:
    writer = csv.DictWriter(file,fieldnames=fieldnames)
    if not file_exists or os.path.getsize(file_path) == 0:
        writer.writeheader()
    writer.writerow(new_data)