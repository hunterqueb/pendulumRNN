import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo

from qutils.integrators import ode87, ode45
from qutils.plot import plotCR3BPPhasePredictions,plotOrbitPredictions, plotSolutionErrors,plot3dCR3BPPredictions,plotStatePredictions,newPlotSolutionErrors
from qutils.mlExtras import findDecAcc
from qutils.mamba import Mamba, MambaConfig
from qutils.ml import trainModel, printModelParmSize, getDevice, Adam_mini, genPlotPrediction, create_datasets, LSTMSelfAttentionNetwork, LSTM
from qutils.tictoc import timer

#import for superweight identification
from qutils.mlExtras import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight

# from nets import Adam_mini

# from memory_profiler import profile

DEBUG = True
plotOn = True
randomIC = False
periodic = True
printoutSuperweight = True
compareLSTM = True

problemDim = 3

sigma = 10
rho = 28
beta = 8/3
t0 = 0; tf = 50

if periodic:
    # sigma = 10
    rho = 350
    # beta = 10
    t0 = 0; tf = 10
parameters = np.array([sigma,rho,beta])

def returnModel(modelString = 'mamba'):
    if modelString == 'mamba':
        model = Mamba(config).to(device).double()
    elif modelString == 'lstm':
        model = LSTMSelfAttentionNetwork(input_size,30,output_size,num_layers,0).double().to(device)
    return model

def lorenzAttractor(t, Y,p=parameters):

    sigma = p[0]
    rho = p[1]
    beta = p[2]

    # Get the position and velocity from the solution vector
    x, y, z = Y[:3]

    # Define the derivative vector

    dydt1 = sigma * (y-x)
    dydt2 = x * (rho - z) - y
    dydt3 = x*y-beta*z

    return np.array([dydt1, dydt2,dydt3])


device = getDevice()


# hyperparameters
n_epochs = 5
lr = 0.001
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = 1


# solve system numerically
np.random.seed()
IC = np.array((1.1, 2, 7))
IC = np.array((0.1,0.1,0.1))
randomized_IC = np.random.uniform(-5, 5, IC.shape)

if randomIC:
    IC = randomized_IC
else:
    IC = np.array((1.1, 2, 7))



delT = 0.001
nSamples = int(np.ceil((tf - t0) / delT))
t = np.linspace(t0, tf, nSamples)

t , numericResult = ode87(lorenzAttractor,[t0,tf],IC,t)

# generate data sets

train_size = 2
test_size = len(t) - train_size

train_in,train_out,test_in,test_out = create_datasets(numericResult,1,train_size,device)
print(train_in)
print(train_out)
loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

config = MambaConfig(d_model=problemDim, n_layers=num_layers)

model = Mamba(config).to(device).double()


optimizer = Adam_mini(model,lr=lr)
criterion = F.smooth_l1_loss

# train with mamba
trainModel(model,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutAcc = True,printOutToc = True)


# plot results
trajPredition = plotStatePredictions(model,t,numericResult,train_in,test_in,train_size,test_size,states=['x','y','z'])

newPlotSolutionErrors(numericResult,trajPredition,t,states=['x','y','z'])

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(numericResult[:,0], numericResult[:,1], numericResult[:,2], 'blue',label = 'Truth')
ax.plot3D(trajPredition[:,0], trajPredition[:,1], trajPredition[:,2], 'green',label = 'Mamba')
ax.set_title(r'Network Prediction of Lorenz Attractor'+'\n'+r'($\sigma$={:.2f}, $\rho$={:.2f}, $\beta$={:.3f})'.format(parameters[0], parameters[1], parameters[2]))
ax.legend()

# plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_title(r'Numerical of Lorenz Attractor'+'\n'+r'($\sigma$={:.2f}, $\rho$={:.2f}, $\beta$={:.3f})'.format(parameters[0], parameters[1], parameters[2]))

printModelParmSize(model)


if printoutSuperweight is True:
    printoutMaxLayerWeight(model)
    getSuperWeight(model)
    plotSuperWeight(model)


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

    trainModel(modelLSTM,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutToc=False)
    networkPredictionLSTM = plotStatePredictions(modelLSTM,t,numericResult,train_in,test_in,train_size,test_size,1,states=('x','y','z'),units=('none','none','none'))
    
    newPlotSolutionErrors(numericResult,networkPredictionLSTM,t,states=['x','y','z'])
    # plotDecAccs(decAcc,t,problemDim)
    errorAvg = np.nanmean(abs(networkPredictionLSTM-numericResult) * 90 / np.pi, axis=0)
    print("Average error of each dimension:")
    unitLabels = ['deg','deg/s','deg','deg/s']
    for i, avg in enumerate(errorAvg, 1):
        print(f"Dimension {i}: {avg} {unitLabels[i-1]}")

    printModelParmSize(modelLSTM)


    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(numericResult[:,0], numericResult[:,1], numericResult[:,2], 'blue',label = 'Truth')
    ax.plot3D(trajPredition[:,0], trajPredition[:,1], trajPredition[:,2], 'green',label = 'Mamba')
    ax.plot3D(networkPredictionLSTM[:,0], networkPredictionLSTM[:,1], networkPredictionLSTM[:,2], 'red',label = "LSTM")
    ax.set_title(r'Network Prediction of Lorenz Attractor'+'\n'+r'($\sigma$={:.2f}, $\rho$={:.2f}, $\beta$={:.3f})'.format(parameters[0], parameters[1], parameters[2]))
    ax.legend()


if plotOn is True:
    plt.show()
