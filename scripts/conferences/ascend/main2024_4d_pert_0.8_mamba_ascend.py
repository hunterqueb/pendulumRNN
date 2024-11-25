
# https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
'''
usually, time seties oreduction is done on a window.
given data from time [t - w,t], you predict t + m where m is any timestep into the future.
w governs how much data you can look at to make a predition, called the look back period.


'''
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import random
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo

from qutils.integrators import myRK4Py, ode45
from qutils.mlExtras import findDecAcc
from qutils.plot import plot3dOrbitPredictions,plotOrbitPhasePredictions, plotSolutionErrors,plotPercentSolutionErrors, plotEnergy,plotStatePredictions
from qutils.orbital import nonDim2Dim4, orbitalEnergy
from qutils.ml import create_datasets, genPlotPrediction, printModelParmSize
from qutils.pinn import PINN,FeedforwardSin,FeedforwardCos
from qutils.tictoc import timer

from qutils.mamba import Mamba, MambaConfig

# seed any random functions
random.seed(123)

# data size set that define amount of data sets we will generate to train the network
DATA_SET_SIZE = 1
TIME_STEP = 0.01

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# ------------------------------------------------------------------------
## NUMERICAL SOLUTION

problemDim = 4 

muR = 3.96800e14
DU = 6378.1e3 # radius of earth in km
TU = ((DU)**3 / muR)**0.5

rLEO = np.array([6.611344740000000e+06,0])
vLEO = np.sqrt(muR/np.linalg.norm(rLEO))
vLEO = np.array([0,vLEO])
TLEO = 2*np.pi * np.sqrt(np.linalg.norm(rLEO)**3 / muR)

# dimensionalized units
mu = 1
r = rLEO/DU
v = vLEO * TU / DU
T = TLEO / TU
pam = mu
a = np.linalg.norm(r)
h = np.cross(r,v)


criterion = F.smooth_l1_loss


config = MambaConfig(d_model=problemDim, n_layers=1)
model = Mamba(config).to(device).double()


muR = 396800
DU = 6378.1 # radius of earth in km
TU = ((DU)**3 / muR)**0.5

p = 20410 # km
e = 0.2

a = p/(1-e**2)

rHEO = np.array([(p/(1+e)),0])
vHEO = np.array([0,np.sqrt(muR*((2/rHEO[0])-(1/a)))])
THEO = 2*np.pi*np.sqrt(a**3/muR)

mu = 1
r = rHEO / DU
v = vHEO * TU / DU
T = THEO / TU

J2 = 1.08263e-3

IC = np.concatenate((r,v))
pam = [mu,J2]

m_sat = 1
c_d = 2.1 #shperical model
A_sat = 1.0013 / (DU ** 2)
h_scale = 50 * 1000 / DU
rho_0 = 1.29 * 1000 ** 2 / (DU**2)

def twoBodyPert(t, y, p=pam):
    r = y[0:2]
    R = np.linalg.norm(r)
    v = y[2:4]
    v_norm = np.linalg.norm(v)

    mu = p[0]; J2 = p[1]
    dydt1 = y[2]
    dydt2 = y[3]

    factor = 1.5 * J2 * (1 / R)**2 / R**3
    j2_accel_x = factor * (1) * r[0]
    j2_accel_y = factor * (3) * r[1]

    rho = rho_0 * np.exp(-R / h_scale)  # Atmospheric density model
    drag_factor = -0.5 * (rho / m_sat) * c_d * A_sat * v_norm
    a_drag_x = drag_factor * y[2]
    a_drag_y = drag_factor *  y[3]

    a_drag_x = 0
    a_drag_y = 0
    j2_accel_x = 0
    j2_accel_y = 0
    dydt3 = -mu / R**3 * y[0] + j2_accel_x + a_drag_x
    dydt4 = -mu / R**3 * y[1] + j2_accel_y + a_drag_y

    return np.array([dydt1, dydt2,dydt3,dydt4])

numPeriods = 2

n_epochs = 10



lr = 0.001
input_size = problemDim
output_size = problemDim
num_layers = 1
p_dropout = 0.0
lookback = 1
p_motion_knowledge = 1/numPeriods

sysfuncptr = twoBodyPert
# sim time
t0, tf = 0, numPeriods * T

t = np.arange(t0, tf, TIME_STEP)

IC = np.concatenate((r,v))

t , numericResult = ode45(sysfuncptr,[t0,tf],IC,t)
t = t * TU
output_seq = numericResult

pertNR = numericResult

train_size = int(len(pertNR) * p_motion_knowledge)
test_size = len(pertNR) - train_size

train_in,train_out,test_in,test_out = create_datasets(pertNR,1,train_size,device)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)


optimizer = torch.optim.Adam(model.parameters(),lr=lr)

for epoch in range(n_epochs):

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


networkPrediction = plotStatePredictions(model,t,output_seq,train_in,test_in,train_size,test_size,DU=DU,TU=TU)
output_seq = nonDim2Dim4(output_seq,DU,TU)

# networkPrediction = nonDim2Dim6(networkPrediction,DU,TU)

plotOrbitPhasePredictions(output_seq,networkPrediction)

plot3dOrbitPredictions(output_seq,networkPrediction)

# networkPrediction = nonDim2Dim6(networkPrediction,DU,TU)
# output_seq = nonDim2Dim6(output_seq,DU,TU)

# plotOrbitPredictions(output_seq,networkPrediction,t=t)
plotSolutionErrors(output_seq,networkPrediction,t)
# plotPercentSolutionErrors(output_seq,networkPrediction,t/tPeriod,semimajorAxis,max(np.linalg.norm(gmatImport[:,3:6],axis=1)))
plotEnergy(output_seq,networkPrediction,t,orbitalEnergy,xLabel='time (TU)',yLabel='Specific Energy')
# plotDecAccs(decAcc,t,problemDim)
errorAvg = np.nanmean(abs(networkPrediction-output_seq), axis=0)
print("Average values of each dimension:")
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg}")

printModelParmSize(model)
torchinfo.summary(model)

# plt.show()

def systemTensor(t, y, p=pam):
    y1 = y[:,0].reshape(-1,1)
    y2 = y[:,1].reshape(-1,1)
    y3 = y[:,2].reshape(-1,1)
    y4 = y[:,3].reshape(-1,1)
    mu = p[0]; J2 = p[1]

    R = torch.norm(y[:,0:2],dim=1).reshape(-1,1)
    V = torch.norm(y[:,2:4],dim=1).reshape(-1,1)

    dydt1 = y3
    dydt2 = y4

    factor = 1.5 * J2 * (1 / R)**2 / R**3
    j2_accel_x = factor * (1) * y1
    j2_accel_y = factor * (3) * y2

    rho = rho_0 * torch.exp(-R / h_scale)  # Atmospheric density model
    drag_factor = -0.5 * (rho / m_sat) * c_d * A_sat * V
    a_drag_x = drag_factor * y3
    a_drag_y = drag_factor *  y4

    dydt3 = -mu / R**3 * y1 + j2_accel_x + a_drag_x
    dydt4 = -mu / R**3 * y2 + j2_accel_y + a_drag_y
    return torch.cat((dydt1,dydt2,dydt3,dydt4),1)

netptr = FeedforwardSin
netptrd = FeedforwardCos

learningRate = 6e-4

desiredSegs = numPeriods * 20

tParts = np.linspace(t0,tf, desiredSegs + 1)

epochs = 10000

numSegs = len(tParts) - 1

epochList = [epochs for _ in range(numSegs)]
learningRateList = [learningRate for _ in range(numSegs)]

dataSet = 10 * 50

pinnSolution = PINN(problemDim,device,netptr,netptrd)
pinnSolution.setupNetwork(1,16,1,learningRateList,dataSet,tParts)
pinnSolution.setupTrialSolution(twoBodyPert,IC)
# pinnSolution.setupConservation(jacobiConst=C0)
# pinnSolution.setPlot(plotCR3BPPhasePredictions)
pinnSolution.setToTrain()

# criterion = torch.nn.MSELoss()
# criterion1 = torch.nn.HuberLoss()

# train each set of networks
timeToTrain = timer()
pinnSolution.trainAnalytical(systemTensor,criterion,epochList)
print('Total Time to Train: {}'.format(timeToTrain.tocVal()))

# set to evaluation
pinnSolution.setToEvaluate()

# evaluate networks
t,yTest = pinnSolution()
t = t * TU
yTruth = pinnSolution.getTrueSolution()

yTruth = nonDim2Dim4(yTruth,DU,TU)
yTest = nonDim2Dim4(yTest,DU,TU)


plotOrbitPhasePredictions(yTruth,yTest)

states = ['x', 'y', 'xdot', 'ydot']
units = ['km', 'km', 'km/s','km/s']
paired_labels = [f'{label} ({unit})' for label, unit in zip(states, units)]

fig, axes = plt.subplots(2,problemDim // 2)

for i, ax in enumerate(axes.flat):
    ax.plot(t, yTruth[:, i], c='b', label='True Motion')
    ax.plot(t, yTest[:, i], c='g', label='Prediction')
    ax.set_xlabel('time (sec)')
    ax.set_ylabel(paired_labels[i])
    ax.grid()

plt.legend(loc='upper left', bbox_to_anchor=(1,0.5))
plt.tight_layout()


plotSolutionErrors(yTruth,yTest,t)
# plotPercentSolutionErrors(output_seq,networkPrediction,t/tPeriod,semimajorAxis,max(np.linalg.norm(gmatImport[:,3:6],axis=1)))
plotEnergy(yTruth,yTest,t,orbitalEnergy,xLabel='Number of Periods (T)',yLabel='Specific Energy')
# plotDecAccs(decAcc,t,problemDim)
errorAvg = np.nanmean(abs(yTest-yTruth), axis=0)
print("Average values of each dimension:")
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg}")

plt.show()
