import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo
from scipy.io import loadmat,savemat

from qutils.integrators import ode85
from qutils.plot import plotCR3BPPhasePredictions,plotOrbitPredictions, plotSolutionErrors,plot3dCR3BPPredictions
from qutils.mlExtras import findDecAcc,generateTrajectoryPrediction
from qutils.orbital import nonDim2Dim6, returnCR3BPIC
from qutils.mamba import Mamba, MambaConfig
from qutils.ml import printModelParmSize, getDevice, Adam_mini
from qutils.tictoc import timer
# from nets import Adam_mini

# from memory_profiler import profile

from nets import create_dataset, LSTMSelfAttentionNetwork

DEBUG = True
plotOn = True

device = getDevice()

dataset = loadmat('satellites-dataset.mat')['dataset'] / 2000

# gridPoints =  np.stack((matrix_t0[:,0:2],matrix_t1[:,0:2]),axis=0)
# gridPoints =  np.stack((matrix_t0[:,2],matrix_t1[:,2]),axis=0)

sequenceLength = dataset.shape[0]
problemDim = dataset.shape[1]


output_seq = np.float64(dataset)
predictionFuture = 10
tf = 33 + predictionFuture
dt = 1
# hyperparameters
n_epochs = 50
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
p_motion_knowledge = 0.75


train_size = int(len(output_seq) * p_motion_knowledge)
# train_size = 2
test_size = len(output_seq) - train_size

train, test = output_seq[:train_size], output_seq[train_size:]

train_in,train_out = create_dataset(train,device,lookback=lookback)
test_in,test_out = create_dataset(test,device,lookback=lookback)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

# initilizing the model, criterion, and optimizer for the data
config = MambaConfig(d_model=problemDim, n_layers=num_layers,d_conv=16)

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
criterion = torch.nn.HuberLoss()

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

model.eval()

# predictionTime = timer()
# with torch.no_grad():
#     train_pred = np.ones_like(output_seq) * np.nan
#     train_pred[lookback:train_size] = model(train_in)[:,-1,:].cpu()

#     test_pred = np.ones_like(output_seq) * np.nan
#     test_pred[train_size+lookback:sequenceLength] = model(test_in)[:, -1, :].cpu()
# predictionTime.toc()

predictionTime = timer()
data_in = train_in
for t in range(int(tf/dt) - len(data_in) + 1):
    with torch.no_grad():
        data_out = model(data_in)
        data_in = torch.cat((data_in,data_out[-1].reshape((1,1,4))),0)
predictionTime.toc()

finalData = data_out[:,-1,:].cpu().numpy() * 2000


# finalData = generateTrajectoryPrediction(train_pred,test_pred) * 2000

# errorAvg = np.nanmean(abs(finalData-output_seq), axis=0)
# print("Average values of each dimension:")
# for i, avg in enumerate(errorAvg, 1):
#     print(f"Dimension {i}: {avg}")

printModelParmSize(model)
torchinfo.summary(model)

dataset = dataset * 2000

dataset = np.append(dataset,np.ones((predictionFuture,problemDim))*np.nan,axis=0)

plt.figure()
plt.plot(finalData[:,0],label="Network Prediction")
plt.plot(dataset[:,0],label="Truth")
plt.title("Active Small Sats")
plt.legend()
plt.figure()
plt.plot(finalData[:,1],label="Network Prediction")
plt.plot(dataset[:,1],label="Truth")
plt.legend()
plt.title("Active Large Sats")
plt.figure()
plt.plot(finalData[:,2],label="Network Prediction")
plt.plot(dataset[:,2],label="Truth")
plt.legend()
plt.title("Small Sat Launches")
plt.figure()
plt.plot(finalData[:,3],label="Network Prediction")
plt.plot(dataset[:,3],label="Truth")
plt.legend()
plt.title("Big Sat Launches")

if plotOn is True:
    plt.show()