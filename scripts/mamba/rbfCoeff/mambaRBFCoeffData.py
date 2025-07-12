import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo
from scipy.io import loadmat,savemat

from qutils.integrators import ode45
from qutils.plot import plotCR3BPPhasePredictions,plotOrbitPredictions, plotSolutionErrors
from qutils.ml.utils import findDecAcc,generateTrajectoryPrediction,getDevice,printModelParmSize
from qutils.orbital import nonDim2Dim4
from qutils.tictoc import timer
from qutils.mambaAtt import Mamba,MambaConfig

from qutils.ml.regression import create_datasets as create_dataset, LSTMSelfAttentionNetwork

DEBUG = True
plotOn = True

device = getDevice()

normalized_coeff = loadmat('matlab/lowerDataMamba/normalized_coeff.mat')['normalized_coeff']
current_coeff = loadmat('matlab/lowerDataMamba/current_coeff.mat')['current_coeff']

# gridPoints =  np.stack((matrix_t0[:,0:2],matrix_t1[:,0:2]),axis=0)
# gridPoints =  np.stack((matrix_t0[:,2],matrix_t1[:,2]),axis=0)
gridPoints =  normalized_coeff.T

sequenceLength = gridPoints.shape[0]
problemDim = gridPoints.shape[1]

dt = 0.1
tf = 7.4

# hyperparameters
n_epochs = 20
# lr = 0.0007
lr = 0.001
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = 1

p_motion_knowledge = 0.4

# train_size = 2
train_size = int(sequenceLength * p_motion_knowledge)
test_size = sequenceLength - train_size


train = gridPoints[:train_size]
test = gridPoints[train_size:]

train_in,train_out = create_dataset(train,device,lookback=lookback)
test_in,test_out = create_dataset(test,device,lookback=lookback)

# testing can be the final matrix

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

# initilizing the model, criterion, and optimizer for the data
# config = MambaConfig(d_model=problemDim, n_layers=num_layers,d_conv=256)
config = MambaConfig(d_model=problemDim, n_layers=num_layers,d_conv=512,expand_factor=1)
model = Mamba(config).to(device).double()

torchinfo.summary(model)

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = F.smooth_l1_loss

trainingTime = timer()
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
        decAcc, err2 = findDecAcc(test_out,y_pred_test,printOut=False)
        err = np.concatenate((err1,err2),axis=0)

    print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))
trainingTime.toc()

model.eval()

predictionTime = timer()
with torch.no_grad():
    train_pred = np.ones_like(gridPoints) * np.nan
    train_pred[lookback:train_size] = model(train_in)[:,-1,:].cpu()

    test_pred = np.ones_like(gridPoints) * np.nan
    test_pred[train_size+lookback:sequenceLength] = model(test_in)[:, -1, :].cpu()
predictionTime.toc()

finalData = generateTrajectoryPrediction(train_pred,test_pred)

scale_factor = np.max(current_coeff)

predictedCoeffNorm = finalData
predictedCoeffScaled = predictedCoeffNorm * scale_factor

trueCoffNorm = normalized_coeff[:,-1]
trueCoffScaled = current_coeff[:,-1]

error = predictedCoeffNorm[-1,:] - trueCoffNorm
errorAvg = np.nanmean(abs(error))
print('Average error in normalized: ',errorAvg)

error = predictedCoeffScaled[-1,:] - trueCoffScaled
errorAvg = np.nanmean(abs(error))
print('Average error in scaled: ',errorAvg)

# # save the final y_pred_test


# # savemat('matlab/lowerDataMamba/prediction/normalized_pdf_tf.mat',{'normalized_pdf_tf': predictedPDFValues})
savemat('matlab/lowerDataMamba/prediction/coeff_prediction.mat',{'normalized_coeff_pred': predictedCoeffNorm,'scaled_coeff_pred': predictedCoeffScaled})