import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo

from qutils.integrators import ode87
from qutils.plot import plotCR3BPPhasePredictions,plotOrbitPredictions, plotSolutionErrors,plotStatePredictions, plot3dCR3BPPredictions,newPlotSolutionErrors
from qutils.ml.utils import getDevice,findDecAcc
from qutils.ml.regression import create_datasets, genPlotPrediction,transferMamba,LSTMSelfAttentionNetwork,transferLSTM, transferModelAll,LSTM
from qutils.ml.superweight import plotSuperWeight, plotMinWeight, printoutMaxLayerWeight
from qutils.orbital import returnCR3BPIC, nonDim2Dim6
from qutils.tictoc import timer
from qutils.ml.mamba import Mamba, MambaConfig

modelString = 'mamba'

modelSaved = False
pretrainedModelPath = 'CR3BP_L4_SP.pth'
plotOn = True

problemDim = 6
m_1 = 5.974E24  # kg
m_2 = 7.348E22 # kg
mu = m_2/(m_1 + m_2)

orbitFamily = 'longPeriod'

CR3BPIC = returnCR3BPIC(orbitFamily,L=4,id=751,stable=True)

x_0,tEnd = CR3BPIC()

IC = np.array(x_0)

DU = 389703
G = 6.67430e-11
TU = 382981


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
t = np.linspace(t0, tf, nSamples)

ODEtime = timer()
t , numericResult = ode87(system,[t0,tf],IC,t)
ODEtime.toc()

output_seq = numericResult

# hyperparameters
n_epochs = 5
# lr = 5*(10**-5)
# lr = 0.85
lr = 0.001
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = 1
# p_motion_knowledge = 0.5
p_motion_knowledge = 1/numPeriods

train_size = int(len(output_seq) * p_motion_knowledge)
test_size = len(output_seq) - train_size

train_in,train_out,test_in,test_out = create_datasets(output_seq,1,train_size,device)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

# initilizing the model, criterion, and optimizer for the data
config = MambaConfig(d_model=problemDim, n_layers=num_layers)

def returnModel(modelString = 'mamba'):
    if modelString == 'mamba':
        model = Mamba(config).to(device).double()
    elif modelString == 'lstm':
        model = LSTM(input_size,30,output_size,num_layers,0).double().to(device)
    return model

model = returnModel(modelString)

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = F.smooth_l1_loss
criterion = torch.nn.HuberLoss()

if not modelSaved:
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

        print("Epoch %d: train loss %f, test loss %f\n" % (epoch, train_loss, test_loss))
    torch.save(model,pretrainedModelPath)
else:
    print('Loading pretrained model: ' + pretrainedModelPath)
    model = torch.load(pretrainedModelPath)

t = t / tEnd

networkPrediction = plotStatePredictions(model,t,output_seq,train_in,test_in,train_size,test_size,DU=DU,TU=TU)
output_seq = nonDim2Dim6(output_seq,DU,TU)

plotCR3BPPhasePredictions(output_seq,networkPrediction,L=None,earth=False,moon=False)
plotCR3BPPhasePredictions(output_seq,networkPrediction,L=None,plane='xz',earth=False,moon=False)
plotCR3BPPhasePredictions(output_seq,networkPrediction,L=None,plane='yz',earth=False,moon=False)
plot3dCR3BPPredictions(output_seq,networkPrediction,L=None,earth=False,moon=False)

newPlotSolutionErrors(output_seq,networkPrediction,t)

errorAvg = np.nanmean(abs(networkPrediction-output_seq), axis=0)
print("Average values of each dimension:")
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg}")





# TRANSFER LEARN

orbitFamily = 'shortPeriod'

CR3BPIC = returnCR3BPIC(orbitFamily,L=4,id=755,stable=True)

x_0,tEnd = CR3BPIC()

IC = np.array(x_0)

numPeriods = 5

t0 = 0; tf = numPeriods * tEnd

delT = 0.001
nSamples = int(np.ceil((tf - t0) / delT))
t = np.linspace(t0, tf, nSamples)

t , numericResult = ode87(system,[t0,tf],IC,t,rtol=1e-15,atol=1e-15)

output_seq = numericResult

n_epochs = 5
# lr = 5*(10**-5)
# lr = 0.85
lr = 0.001
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = 1
# p_motion_knowledge = 0.5
p_motion_knowledge = 1/numPeriods

train_size = int(len(output_seq) * p_motion_knowledge)
test_size = len(output_seq) - train_size

train_in,train_out,test_in,test_out = create_datasets(output_seq,1,train_size,device)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

# initilizing the model, criterion, and optimizer for the data
config = MambaConfig(d_model=problemDim, n_layers=num_layers)
newModel = Mamba(config).to(device).double()

newModel = returnModel(modelString)

if modelString == "mamba":
    newModel = transferMamba(model,newModel,[True,True,False])
else:
    newModel = transferLSTM(model,newModel)

# newModel = transferModelAll(model,newModel)

# newModel = LSTMSelfAttentionNetwork(input_size,50,output_size,num_layers,0).double().to(device)

optimizer = torch.optim.Adam(newModel.parameters(),lr=lr)
criterion = F.smooth_l1_loss

for epoch in range(n_epochs):

    # trajPredition = plotPredition(epoch,model,'target',t=t*TU,output_seq=pertNR)

    newModel.train()
    for X_batch, y_batch in loader:
        y_pred = newModel(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    newModel.eval()
    with torch.no_grad():
        y_pred_train = newModel(train_in)
        train_loss = np.sqrt(criterion(y_pred_train, train_out).cpu())
        y_pred_test = newModel(test_in)
        test_loss = np.sqrt(criterion(y_pred_test, test_out).cpu())

        decAcc, err1 = findDecAcc(train_out,y_pred_train,printOut=False)
        decAcc, err2 = findDecAcc(test_out,y_pred_test)
        err = np.concatenate((err1,err2),axis=0)

    print("Epoch %d: train loss %f, test loss %f\n" % (epoch, train_loss, test_loss))

t = t / tEnd
networkPrediction = plotStatePredictions(model,t,output_seq,train_in,test_in,train_size,test_size,DU=DU,TU=TU)
output_seq = nonDim2Dim6(output_seq,DU,TU)

plotCR3BPPhasePredictions(output_seq,networkPrediction,L=None,earth=False,moon=False)
plotCR3BPPhasePredictions(output_seq,networkPrediction,L=None,plane='xz',earth=False,moon=False)
plotCR3BPPhasePredictions(output_seq,networkPrediction,L=None,plane='yz',earth=False,moon=False)
plot3dCR3BPPredictions(output_seq,networkPrediction,L=None,earth=False,moon=False)

newPlotSolutionErrors(output_seq,networkPrediction,t)

errorAvg = np.nanmean(abs(networkPrediction-output_seq), axis=0)
print("Average values of each dimension:")
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg}")

torchinfo.summary(model)
torchinfo.summary(newModel)

printoutMaxLayerWeight(model)
printoutMaxLayerWeight(newModel)

plt.figure()
plotSuperWeight(model,newPlot=False)
plotSuperWeight(newModel,newPlot=False)
plt.grid()
plt.tight_layout()

plt.figure()
plotMinWeight(model,newPlot=False)
plotMinWeight(newModel,newPlot=False)
plt.grid()
plt.tight_layout()


if plotOn is True:
    plt.show()

