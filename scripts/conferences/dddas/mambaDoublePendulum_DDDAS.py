import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data

from qutils.integrators import ode45
from qutils.plot import plotCR3BPPhasePredictions,plotOrbitPredictions, plotSolutionErrors, plotStatePredictions

from qutils.ml import printModelParmSize, getDevice, create_datasets, genPlotPrediction, trainModel, LSTMSelfAttentionNetwork,Adam_mini
from qutils.mlExtras import findDecAcc
from qutils.orbital import nonDim2Dim4

from qutils.mamba import Mamba, MambaConfig

from qutils.mlExtras import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight

plotOn = True
printoutSuperweight = True
compareLSTM = True
periodic = True

problemDim = 4 

device = getDevice()

m1 = 1
m2 = m1
l1 = 1
l2 = l1
g = 9.81
parameters = np.array([m1,m2,l1,l2,g])

def returnModel(modelString = 'mamba'):
    if modelString == 'mamba':
        model = Mamba(config).to(device).double()
    elif modelString == 'lstm':
        model = LSTMSelfAttentionNetwork(input_size,30,output_size,num_layers,0).double().to(device)
    return model


def doublePendulumODE(t,y,p=parameters):
    # p = [m1,m2,l1,l2,g]
    m1 = p[0]
    m2 = p[1]
    l1 = p[2]
    l2 = p[3]
    g = p[4]

    theta1 = y[0]
    theta2 = y[2]

    dydt1 = y[1] #theta1dot

    dydt2 = (m2*g*np.sin(theta2)*np.cos(theta1-theta2) - m2*np.sin(theta1-theta2)*(l1*y[1]**2*np.cos(theta1-theta2) + l2*y[3]**2)
            - (m1+m2)*g*np.sin(theta1)) / l1 / (m1 + m2*np.sin(theta1-theta2)**2) #theta1ddot

    dydt3 = y[3] #theta2dot

    dydt4 = ((m1+m2)*(l1*y[1]**2*np.sin(theta1-theta2) - g*np.sin(theta2) + g*np.sin(theta1)*np.cos(theta1-theta2))
            + m2*l2*y[3]**2*np.sin(theta1-theta2)*np.cos(theta1-theta2)) / l2 / (m1 + m2*np.sin(theta1-theta2)**2) #theta2ddot

    return np.array((dydt1,dydt2,dydt3,dydt4))

if periodic:
    theta1_0 = np.radians(10)
    theta2_0 = np.radians(13)
    thetadot1_0 = np.radians(0)
    thetadot2_0 = np.radians(0)
else:    
    theta1_0 = np.radians(80)
    theta2_0 = np.radians(135)
    thetadot1_0 = np.radians(-1)
    thetadot2_0 = np.radians(0.7)

initialConditions = np.array([theta1_0,thetadot1_0,theta2_0,thetadot2_0],dtype=np.float64)

# initialConditions = np.radians(np.random.uniform(-180, 180, (problemDim,)))

tStart = 0
tEnd = 20
tSpan = np.array([tStart,tEnd])
dt = 0.01
tSpanExplicit = np.linspace(tStart,tEnd,int(tEnd / dt))

t , numericResult = ode45(doublePendulumODE,[tStart,tEnd],initialConditions,tSpanExplicit)

output_seq = numericResult

# hyperparameters
n_epochs = 5
# lr = 5*(10**-5)
# lr = 0.85
lr = 0.8
lr = 0.08
lr = 0.004
lr = 0.0001
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = 1

train_size = 2
test_size = len(output_seq) - train_size

train_in,train_out,test_in,test_out = create_datasets(output_seq,1,train_size,device)
print(train_in)
print(train_out)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

# initilizing the model, criterion, and optimizer for the data
config = MambaConfig(d_model=problemDim, n_layers=num_layers)
model = Mamba(config).to(device).double()
# model = LSTM(input_size,10,output_size,num_layers,0).double().to(device)

optimizer = Adam_mini(model,lr=lr)
criterion = F.smooth_l1_loss


trainModel(model,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutToc=False)

networkPrediction = plotStatePredictions(model,t,output_seq,train_in,test_in,train_size,test_size,1,states=('$\\theta_1$','$\\theta_2$','$\dot{\\theta_1}$','$\dot{\\theta_2}$'),units=('rad','rad','rad/s','rad/s'))

plotSolutionErrors(output_seq,networkPrediction,t,units='rad',states=('\\theta_1','\\theta_2'))
# plotDecAccs(decAcc,t,problemDim)
errorAvg = np.nanmean(abs(networkPrediction-output_seq) * 90 / np.pi, axis=0)
print("Average error of each dimension:")
unitLabels = ['deg','deg/s','deg','deg/s']
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg} {unitLabels[i-1]}")

printModelParmSize(model)

if printoutSuperweight is True:
    printoutMaxLayerWeight(model)
    getSuperWeight(model)
    plotSuperWeight(model)


if compareLSTM is True:
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
    networkPredictionLSTM = plotStatePredictions(modelLSTM,t,output_seq,train_in,test_in,train_size,test_size,1,states=('$\\theta_1$','$\\theta_2$','$\dot{\\theta_1}$','$\dot{\\theta_2}$'),units=('rad','rad','rad/s','rad/s'))
    
    plotSolutionErrors(output_seq,networkPredictionLSTM,t,units='rad',states=('\\theta_1','\\theta_2'))
    # plotDecAccs(decAcc,t,problemDim)
    errorAvg = np.nanmean(abs(networkPrediction-output_seq) * 90 / np.pi, axis=0)
    print("Average error of each dimension:")
    unitLabels = ['deg','deg/s','deg','deg/s']
    for i, avg in enumerate(errorAvg, 1):
        print(f"Dimension {i}: {avg} {unitLabels[i-1]}")

    printModelParmSize(modelLSTM)


    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(output_seq[:,0],output_seq[:,1],label = "Truth")
    plt.plot(networkPrediction[:,0],networkPrediction[:,1],label = "NN")
    plt.plot(networkPredictionLSTM[:,0],networkPredictionLSTM[:,1],label = "NN")

    plt.xlabel('$\\theta_1$')
    plt.ylabel('$\dot{\\theta_1}$')
    plt.axis('equal')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(output_seq[:,2],output_seq[:,3],label = "Truth")
    plt.plot(networkPrediction[:,2],networkPrediction[:,3],label = "Mamba")
    plt.plot(networkPredictionLSTM[:,2],networkPredictionLSTM[:,3],label = "LSTM")
    plt.xlabel('$\\theta_2$')
    plt.ylabel('$\dot{\\theta_2}$')
    plt.axis('equal')
    plt.legend()
    plt.grid()

else:
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(output_seq[:,0],output_seq[:,1],'r',label = "Truth")
    plt.plot(networkPrediction[:,0],networkPrediction[:,1],'b',label = "NN")
    plt.xlabel('$\\theta_1$')
    plt.ylabel('$\dot{\\theta_1}$')
    plt.axis('equal')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(output_seq[:,2],output_seq[:,3],'r',label = "Truth")
    plt.plot(networkPrediction[:,2],networkPrediction[:,3],'b',label = "NN")
    plt.xlabel('$\\theta_2$')
    plt.ylabel('$\dot{\\theta_2}$')
    plt.axis('equal')
    plt.legend()
    plt.grid()



if plotOn is True:
    plt.show()
