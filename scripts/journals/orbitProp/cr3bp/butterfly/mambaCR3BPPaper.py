import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import torch
import torch.nn.functional as F
import torchinfo
from scipy.io import loadmat

from qutils.integrators import ode87
from qutils.plot import plotCR3BPPhasePredictions,plotOrbitPredictions, plotSolutionErrors,plot3dCR3BPPredictions,plotStatePredictions, plotEnergy
from qutils.ml.utils import findDecAcc,printModelParmSize, getDevice, Adam_mini
from qutils.orbital import nonDim2Dim6, dim2NonDim6, returnCR3BPIC, jacobiConstant6
from qutils.ml.mamba import Mamba, MambaConfig
from qutils.ml.regression import trainModel, genPlotPrediction, create_datasets,LSTMSelfAttentionNetwork
from qutils.tictoc import timer
# from nets import Adam_mini

# from memory_profiler import profile
from qutils.ml.superweight import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight,findMambaSuperActivation, plotSuperActivation

DEBUG = True
plotOn = False
printoutSuperweight = True
compareLSTM = False
saveData = False
percentRMSE = True

problemDim = 6
m_1 = 5.974E24  # kg
m_2 = 7.348E22 # kg
mu = m_2/(m_1 + m_2)

# halo orbit around L1 - id 754

# halo around l3 - id 10

# butterfly id 270

# dragonfly id 71

# lyapunov id 312

orbitFamily = 'butterfly'

CR3BPIC = returnCR3BPIC(orbitFamily,L="north",id=1080)

# orbitFamily = 'longPeriod'

# CR3BPIC = returnCR3BPIC("shortPeriod",L=4,id=806)

x_0,tEnd = CR3BPIC()

IC = np.array(x_0)

print(IC)
print(tEnd)
def system(t, Y,mu=mu):
    """Solve the CR3BP in nondimensional coordinates.
    
    The state vector is Y, with the first three components as the
    position of $m$, and the second three components its velocity.
    
    The solution is parameterized on $\\pi_2$, the mass ratio.
    """
    # Get the position and velocity from the solution vector
    x, y, z = Y[:3]
    xdot, ydot, zdot = Y[3:]

    # Define the derivative vector

    dydt1 = xdot
    dydt2 = ydot
    dydt3 = zdot

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)

    dydt4 = 2 * ydot + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    dydt5 = -2 * xdot + y - (1 - mu) * y / r1**3 - mu * y / r2**3
    dydt6 = -(1 - mu) * z / r1**3 - mu * z / r2**3

    return np.array([dydt1, dydt2,dydt3,dydt4,dydt5,dydt6])



device = getDevice()

numPeriods = 5

t0 = 0; tf = numPeriods * tEnd

delT = 0.001
nSamples = int(np.ceil((tf - t0) / delT))

# t = np.linspace(t0, tf, nSamples)

# # t , numericResult = ode1412(system,[t0,tf],IC,t)
# t , numericResult = ode87(system,[t0,tf],IC,t,rtol=1e-15,atol=1e-15)
# output_seq = numericResult


from scipy.io import loadmat
cr3bp = loadmat("scripts/journals/orbitProp/cr3bp/butterfly/CR3BP_butterfly_1080.mat")

output_seq = cr3bp['XODE']
t = cr3bp['tResult']

t = t / tEnd


# hyperparameters
n_epochs = 5
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
# p_motion_knowledge = 0.5
p_motion_knowledge = 1/numPeriods


train_size = int(len(output_seq) * p_motion_knowledge)
# train_size = 2
test_size = len(output_seq) - train_size

print(train_size)
print(test_size)

train_in,train_out,test_in,test_out = create_datasets(output_seq,1,train_size,device)

# initilizing the model, criterion, and optimizer for the data
config = MambaConfig(d_model=problemDim, n_layers=num_layers,d_conv=16)

def returnModel(modelString = 'mamba'):
    if modelString == 'mamba':
        model = Mamba(config).to(device).double()
    elif modelString == 'lstm':
        model = LSTMSelfAttentionNetwork(input_size,30,output_size,num_layers,0).double().to(device)
    return model

model = returnModel()

optimizer = Adam_mini(model,lr=lr)
# optimizer = Adam_mini(model,lr=lr)

criterion = F.smooth_l1_loss
# criterion = torch.nn.HuberLoss()

timeToTrain = trainModel(model,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutAcc = True,printOutToc = True)

DU = 389703
G = 6.67430e-11
# TU = np.sqrt(DU**3 / (G*(m_1+m_2)))
TU = 382981

networkPrediction, testTime = plotStatePredictions(model,t,output_seq,train_in,test_in,train_size,test_size,DU=DU,TU=TU,outputToc = True)
output_seq = nonDim2Dim6(output_seq,DU,TU)
print(output_seq[0,:])
plotCR3BPPhasePredictions(output_seq,networkPrediction,L=2,DU=DU)
plotCR3BPPhasePredictions(output_seq,networkPrediction,L=2,plane='xz',DU=DU)
plotCR3BPPhasePredictions(output_seq,networkPrediction,L=2,plane='yz',DU=DU)
# plot3dCR3BPPredictions(output_seq,networkPrediction,L=None,earth=False,moon=False)
plot3dCR3BPPredictions(output_seq,networkPrediction,earth=False,networkLabel="Mamba",DU=DU,L=None,moon=False)
plt.plot(-mu * DU, 0, 0, 'ko', label='Earth')
plt.plot((1-mu) * DU, 0, 0, 'go', label='Moon')
plt.legend(fontsize=10)
plt.tight_layout()

from qutils.plot import newPlotSolutionErrors
newPlotSolutionErrors(output_seq,networkPrediction,t,timeLabel="Orbit Periods")

from qutils.ml.utils import rmse
rmseMamba = rmse(output_seq,networkPrediction,percentRMSE=True)

errorAvg = np.nanmean(abs(networkPrediction-output_seq), axis=0)
print("Average values of each dimension:")
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg}")

printModelParmSize(model)
torchinfo.summary(model)

if printoutSuperweight:
    printoutMaxLayerWeight(model)
    getSuperWeight(model)
    plotSuperWeight(model)

    magnitude, index = findMambaSuperActivation(model,test_in)
    plotSuperActivation(magnitude, index)

    import csv
    import os

    fieldnames = ["in_proj","conv1d","x_proj","dt_proj","out_proj"]
    new_data_mamba = {"in_proj":magnitude[0].norm().item(),"conv1d":magnitude[1].norm().item(),"x_proj":magnitude[2].norm().item(),"dt_proj":magnitude[3].norm().item(),"out_proj":magnitude[4].norm().item()}

    file_path = 'cr3bpSW.csv'
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file,fieldnames=fieldnames)
        if not file_exists or os.path.getsize(file_path) == 0:
            writer.writeheader()
        writer.writerow(new_data_mamba)

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

    networkPredictionLSTM, testTimeLSTM = plotStatePredictions(modelLSTM,t,output_seq,train_in,test_in,train_size,test_size,DU=DU,TU=TU,outputToc = True)
    output_seq = nonDim2Dim6(output_seq,DU,TU)

    plot3dCR3BPPredictions(output_seq,networkPrediction,earth=False,networkLabel="Mamba",DU=DU,L=None,moon=False)
    plt.plot(networkPredictionLSTM[:, 0], networkPredictionLSTM[:, 1], networkPredictionLSTM[:, 2], label='LSTM',linestyle='dashed')
    plt.plot(-mu * DU, 0, 0, 'ko', label='Earth')
    plt.plot((1-mu) * DU, 0, 0, 'go', label='Moon')
    plt.legend(fontsize=10)
    plt.tight_layout()

    plotCR3BPPhasePredictions(output_seq,networkPrediction,networkLabel="Mamba",L=None,DU=DU,earth=False,moon=False)
    plt.plot(networkPredictionLSTM[:, 0], networkPredictionLSTM[:, 1], label='LSTM',linestyle='dashed')
    plt.plot(-mu * DU, 0, 'ko', label='Earth')
    plt.plot((1-mu) * DU, 0, 'go', label='Moon')
    plt.legend()

    plotCR3BPPhasePredictions(output_seq,networkPredictionLSTM,plane='xz')
    plotCR3BPPhasePredictions(output_seq,networkPredictionLSTM,plane='yz')

    newPlotSolutionErrors(output_seq,networkPredictionLSTM,t,timeLabel="Orbit Periods")

    fig, axes = newPlotSolutionErrors(output_seq,networkPredictionLSTM,t,timeLabel="Orbit Periods",percentError=True,states = ['x', 'y', 'z', '$\dot{x}$', '$\dot{y}$', '$\dot{z}$'])
    newPlotSolutionErrors(output_seq,networkPrediction,t,timeLabel="Orbit Periods",newPlot=axes,networkLabels=["LSTM","Mamba"],percentError=True,states = ['x', 'y', 'z', '$\dot{x}$', '$\dot{y}$', '$\dot{z}$'])
    mambaLine = mlines.Line2D([], [], color='b', label='LSTM')
    LSTMLine = mlines.Line2D([], [], color='orange', label='Mamba')
    fig.legend(handles=[mambaLine,LSTMLine])
    # fig.tight_layout()
    fig.set_size_inches(12, 8)  # Adjust the figure size here (width, height)

    # plotPercentSolutionErrors(output_seq,networkPredictionLSTM,t,semimajorAxis,max(np.linalg.norm(gmatImport[:,3:6],axis=1)))

    plotEnergy(output_seq,networkPrediction,t,jacobiConstant6,xLabel='Number of Periods (T)',yLabel='Jacobi Constant',nonDim=dim2NonDim6,DU = DU, TU = TU,networkLabel="Mamba")
    plt.plot(t,jacobiConstant6(dim2NonDim6(networkPredictionLSTM,DU=DU,TU=TU)),label='LSTM',linestyle='dashed')
    plt.legend()

    plotEnergy(output_seq,networkPrediction,t,jacobiConstant6,xLabel='Number of Periods (T)',yLabel='Jacobi Constant',nonDim=dim2NonDim6,DU = DU, TU = TU,networkLabel="Mamba")
    plt.legend(loc="lower left")

    rmseLSTM = rmse(output_seq,networkPredictionLSTM,percentRMSE=True)

    errorAvg = np.nanmean(abs(networkPredictionLSTM-output_seq), axis=0)
    print("Average values of each dimension:")
    for i, avg in enumerate(errorAvg, 1):
        print(f"Dimension {i}: {avg}")

    printModelParmSize(modelLSTM)
    torchinfo.summary(modelLSTM)


if plotOn is True:
    plt.show()

if saveData is True:
    import csv
    import os

    fieldnames = ["x","y","z","vx","vy","vz"]
    new_data_mamba = {"x":rmseMamba[0],"y":rmseMamba[1],"z":rmseMamba[2],"vx":rmseMamba[3],"vy":rmseMamba[4],"vz":rmseMamba[5]}
    new_data_LSTM = {"x":rmseLSTM[0],"y":rmseLSTM[1],"z":rmseLSTM[2],"vx":rmseLSTM[3],"vy":rmseLSTM[4],"vz":rmseLSTM[5]}


    file_path = 'cr3bpRMSEMamba.csv'
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file,fieldnames=fieldnames)
        if not file_exists or os.path.getsize(file_path) == 0:
            writer.writeheader()
        writer.writerow(new_data_mamba)

    file_path = 'cr3bpRMSELSTM.csv'
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file,fieldnames=fieldnames)
        if not file_exists or os.path.getsize(file_path) == 0:
            writer.writeheader()
        writer.writerow(new_data_LSTM)


    fieldnames = ["Mamba Train","LSTM Train","Mamba Test","LSTM Test"]
    new_data = {"Mamba Train":timeToTrain,"LSTM Train":timeToTrainLSTM,"Mamba Test":testTime,"LSTM Test":testTimeLSTM}


    file_path = 'cr3bpTime.csv'
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file,fieldnames=fieldnames)
        if not file_exists or os.path.getsize(file_path) == 0:
            writer.writeheader()
        writer.writerow(new_data)
