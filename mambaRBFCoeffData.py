import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo
from scipy.io import loadmat,savemat

from qutils.integrators import ode45
from qutils.plot import plotCR3BPPhasePredictions,plotOrbitPredictions, plotSolutionErrors
from qutils.mlExtras import findDecAcc
from qutils.orbital import nonDim2Dim4
from qutils.tictoc import timer

from nets import create_dataset, LSTMSelfAttentionNetwork
from mamba import Mamba, MambaConfig

DEBUG = True
plotOn = True

is_cuda = torch.cuda.is_available()
# torch.backends.mps.is_available() checks for metal support, used in nightly build so handled expection incase its run on different version
try:
    is_mps = torch.backends.mps.is_available()
    is_mps = False
except:
    is_mps = False
# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
elif is_mps:
    device = torch.device("mps")
    print('Metal GPU is available')
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

normalized_coeff = loadmat('matlab/lowerDataMamba/normalized_coeff.mat')['normalized_coeff']

# gridPoints =  np.stack((matrix_t0[:,0:2],matrix_t1[:,0:2]),axis=0)
# gridPoints =  np.stack((matrix_t0[:,2],matrix_t1[:,2]),axis=0)
gridPoints =  normalized_coeff

problemDim = gridPoints.shape[0]
sequenceLength = gridPoints.shape[1]

dt = 0.1
tf = 7.4

# hyperparameters
n_epochs = 100
# lr = 0.0007
lr = 0.001
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = 1

p_motion_knowledge = 0.1

# train_size = 2
train_size = int(sequenceLength * p_motion_knowledge)
test_size = sequenceLength - train_size


train = gridPoints[:,:train_size].T
test = gridPoints[:,train_size:].T

train_in,train_out = create_dataset(train,device,lookback=lookback)
test_in,test_out = create_dataset(test,device,lookback=lookback)

# testing can be the final matrix

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

# initilizing the model, criterion, and optimizer for the data
# config = MambaConfig(d_model=problemDim, n_layers=num_layers,d_conv=256)
config = MambaConfig(d_model=problemDim, n_layers=num_layers,d_conv=1024,expand_factor=1)
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
data_in = train_in

predictionTime = timer()
for t in range(int(tf/dt) + 1):
    with torch.no_grad():
        data_out = model(data_in)
        data_in = data_out
predictionTime.toc()

# predictedPDFValues = data_out.cpu().numpy()
# PDFValues = matrix_tf[:,2]
# error = predictedPDFValues - PDFValues
# errorAvg = np.mean(abs(error))
# print('Average error: ',errorAvg)
# # save the final y_pred_test


# # savemat('matlab/lowerDataMamba/prediction/normalized_pdf_tf.mat',{'normalized_pdf_tf': predictedPDFValues})
# savemat('matlab/lowerDataMamba/prediction/normalized_coeff_tf.mat',{'normalized_coeff_tf': predictedPDFValues})