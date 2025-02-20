import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import latex

from qutils.integrators import ode45
from qutils.ml import printModelParmSize, getDevice, create_datasets, genPlotPrediction, trainModel

from qutils.mamba import Mamba, MambaConfig

from qutils.mlExtras import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight

from qutils.mlSuperweight import findMambaSuperActivation,plotSuperActivation,zeroModelWeight

activationArea = 'output'
layer_path = "layers"
DEBUG = False

mambaLayerAttributes = ["in_proj","conv1d","x_proj","dt_proj","out_proj"]

plotOn = True
printoutSuperweight = True

problemDim = 4 

device = getDevice()

m1 = 1
m2 = m1
l1 = 1
l2 = l1
g = 9.81
parameters = np.array([m1,m2,l1,l2,g])

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
p_motion_knowledge = 0.5


train_size = int(len(output_seq) * p_motion_knowledge)
# train_size = 2
test_size = len(output_seq) - train_size
train_in,train_out,test_in,test_out = create_datasets(output_seq,1,train_size,device)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

# initilizing the model, criterion, and optimizer for the data
config = MambaConfig(d_model=problemDim, n_layers=num_layers)
model = Mamba(config).to(device).double()
# model = LSTM(input_size,10,output_size,num_layers,0).double().to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = F.smooth_l1_loss

trainModel(model,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutToc=False)

# zeroModelWeight(model,"dt_proj","weight")
# zeroModelWeight(model,"dt_proj","bias")
# zeroModelWeight(model,"in_proj","weight")

# state_dict = model.state_dict()

# print(state_dict["layers.0.mixer.dt_proj.weight"])
# print(state_dict["layers.0.mixer.dt_proj.bias"])
# print(state_dict["layers.0.mixer.conv1d.weight"])


print(model(torch.zeros_like(test_in,device=device).double()))
from qutils.plot import plotStatePredictions

networkPrediction = plotStatePredictions(model,t,output_seq,train_in,test_in,train_size,test_size,1,states=('$\\theta_1$','$\\theta_2$','$\dot{\\theta_1}$','$\dot{\\theta_2}$'),units=('rad','rad','rad/s','rad/s'),plotOn=not DEBUG)


# spikes_input = [i for i, value in enumerate(magnitude) if abs(value.norm()) > 50]
# print(f"Activation spikes")
# for i in spikes_input:
#     spike_index = index[i]
#     print(f" - layer {i}, value {magnitude[i]}, index {tuple(i.item() for i in spike_index)}")


# plotDecAccs(decAcc,t,problemDim)
errorAvg = np.nanmean(abs(networkPrediction-output_seq) * 90 / np.pi, axis=0)
print("Average error of each dimension:")
unitLabels = ['deg','deg/s','deg','deg/s']
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg} {unitLabels[i-1]}")

printModelParmSize(model)



magnitude, index = findMambaSuperActivation(model,test_in)

if plotOn is True:
    plotSuperActivation(magnitude,index)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(output_seq[:,0],output_seq[:,1],'r',label = "Truth")
    plt.plot(networkPrediction[:,0],networkPrediction[:,1],'b',label = "NN")
    plt.xlabel('Theta 1')
    plt.ylabel('Theta 1 Dot')
    plt.axis('equal')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(output_seq[:,2],output_seq[:,3],'r',label = "Truth")
    plt.plot(networkPrediction[:,2],networkPrediction[:,3],'b',label = "NN")
    plt.xlabel('Theta 2')
    plt.ylabel('Theta 2 Dot')
    plt.axis('equal')
    plt.legend()
    plt.grid()

    plt.show()


