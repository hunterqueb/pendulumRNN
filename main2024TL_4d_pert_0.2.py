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

from qutils.integrators import myRK4Py, ode45
from qutils.mlExtras import findDecAcc
from qutils.plot import plotOrbitPhasePredictions
from qutils.orbital import nonDim2Dim4

from nets import LSTMSelfAttentionNetwork, create_dataset, LSTM, transferLSTM

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

mu = 1
r = rLEO/DU
v = vLEO * TU / DU
T = TLEO / TU
pam = mu
a = np.linalg.norm(r)
h = np.cross(r,v)

def twoBodyCirc(t, y, p=pam):
    r = y[0:2]
    R = np.linalg.norm(r)

    dydt1 = y[2]
    dydt2 = y[3]

    dydt3 = -p / R**3 * y[0]
    dydt4 = -p / R**3 * y[1]

    return np.array([dydt1, dydt2,dydt3,dydt4])


sysfuncptr = twoBodyCirc
# sim time
t0, tf = 0, 5*T

t = np.arange(t0, tf, TIME_STEP)
degreesOfFreedom = problemDim
# initilize the arrays used to store the info from the numerical solution
IC = np.zeros((degreesOfFreedom,DATA_SET_SIZE))
output_seq = np.zeros((len(t),degreesOfFreedom))

# generate random data set of input thetas and output thetas and theta dots over a time series
IC = np.concatenate((r,v))
numericResult = myRK4Py(sysfuncptr,IC,t,paramaters=pam)
_ , numericResult = ode45(sysfuncptr,[t0,tf],IC,t)

output_seq = numericResult

circNR = numericResult

# hyperparameters
n_epochs = 1
# lr = 5*(10**-5)
# lr = 0.85
lr = 0.8
lr = 0.08
lr = 0.004
lr = 0.001
input_size = degreesOfFreedom
output_size = degreesOfFreedom
num_layers = 1
hidden_size = 50
p_dropout = 0.0
lookback = 1
p_motion_knowledge = 0.2

'''
with a lookback window of 4, the input to the network is [t,t+1,t+2,t+3] and the output is [t+1,t+2,t+3,t+4]

for long time series, overlapping windows can be created.
A time series of n time steps can produce roughly n windows because a window can start from any time step as long as the window does 
not go beyond the boundary of the time series. Within one window, there are multiple consecutive time steps of values. 
In each time step, there can be multiple features.
'''


train_size = int(len(output_seq) * p_motion_knowledge)
test_size = len(output_seq) - train_size

train, test = output_seq[:train_size], output_seq[train_size:]

train_in,train_out = create_dataset(train,device,lookback=lookback)
test_in,test_out = create_dataset(test,device,lookback=lookback)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

# initilizing the model, criterion, and optimizer for the data
model = LSTMSelfAttentionNetwork(input_size,hidden_size,output_size,num_layers, p_dropout).double().to(device)
# model = LSTM(input_size,hidden_size,output_size,num_layers, p_dropout).double().to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = F.smooth_l1_loss
# criterion = torch.nn.MSELoss()
# Define the mean absolute error (MAE) loss function
# mae_loss = F.l1_loss(predicted, target)

# Define the Huber loss function with delta=1.0
# huber_loss = F.smooth_l1_loss(predicted, target, reduction='mean', delta=1.0)

def plotPredition(epoch,model,prediction='source',err=None,t = t * TU,output_seq = circNR):
        with torch.no_grad():
            # shift train predictions for plotting
            train_plot = np.ones_like(output_seq) * np.nan
            y_pred = model(train_in)
            y_pred = y_pred[:, -1, :]
            train_plot[lookback:train_size] = model(train_in)[:, -1, :].cpu()
            # shift test predictions for plotting
            test_plot = np.ones_like(output_seq) * np.nan
            test_plot[train_size+lookback:len(output_seq)] = model(test_in)[:, -1, :].cpu()

        output_seq = nonDim2Dim4(output_seq)
        train_plot = nonDim2Dim4(train_plot)
        test_plot = nonDim2Dim4(test_plot)

        fig, axes = plt.subplots(2,2)

        axes[0,0].plot(t,output_seq[:,0], c='b',label = 'True Motion')
        axes[0,0].plot(t,train_plot[:,0], c='r',label = 'Training Region')
        axes[0,0].plot(t,test_plot[:,0], c='g',label = 'Predition')
        axes[0,0].set_xlabel('time (sec)')
        axes[0,0].set_ylabel('x (km)')

        axes[0,1].plot(t,output_seq[:,1], c='b',label = 'True Motion')
        axes[0,1].plot(t,train_plot[:,1], c='r',label = 'Training Region')
        axes[0,1].plot(t,test_plot[:,1], c='g',label = 'Predition')
        axes[0,1].set_xlabel('time (sec)')
        axes[0,1].set_ylabel('y (km)')

        axes[1,0].plot(t,output_seq[:,2], c='b',label = 'True Motion')
        axes[1,0].plot(t,train_plot[:,2], c='r',label = 'Training Region')
        axes[1,0].plot(t,test_plot[:,2], c='g',label = 'Predition')
        axes[1,0].set_xlabel('time (sec)')
        axes[1,0].set_ylabel('xdot (km/s)')

        axes[1,1].plot(t,output_seq[:,3], c='b',label = 'True Motion')
        axes[1,1].plot(t,train_plot[:,3], c='r',label = 'Training Region')
        axes[1,1].plot(t,test_plot[:,3], c='g',label = 'Predition')
        axes[1,1].set_xlabel('time (sec)')
        axes[1,1].set_ylabel('ydot (km/s)')



        plt.legend(loc='upper left', bbox_to_anchor=(1,0.5))
        plt.tight_layout()

        if prediction == 'source':
            plt.savefig('predict/predict%d.png' % epoch)
        if prediction == 'target':
            plt.savefig('predict/newPredict%d.png' % epoch)
        plt.close()

        if err is not None:
            fig, (ax1, ax2) = plt.subplots(2,1)
            ax1.plot(err[:,0:2],label=('x','y'))
            ax1.set_xlabel('node #')
            ax1.set_ylabel('error (km)')
            ax1.legend()
            ax2.plot(err[:,2:4],label=('xdot','ydot'))
            ax2.set_xlabel('node #')
            ax2.set_ylabel('error (km/s)')
            ax2.legend()
            # ax2.plot(np.average(err,axis=0)*np.ones(err.shape))
            plt.show()
        # filter out nan values for better post processing
        train_plot = train_plot[~np.isnan(train_plot)]
        test_plot = test_plot[~np.isnan(test_plot)]

        trajPredition = np.concatenate((train_plot,test_plot))

        return trajPredition.reshape((len(trajPredition),1))

for epoch in range(n_epochs):

    trajPredition = plotPredition(epoch,model)

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

err = nonDim2Dim4(err)
plotPredition(epoch+1,model,err = err)

# """
# TRANSFER LEARN TO NEW, NONLINEAR SYSTEM ON DIFFERENT INITIAL CONDITIONS AND DIFFERENT TIME PERIOD AND DIFFERENT TIME STEP
# """

TIME_STEP = 0.005

# transfer to different system

newModel = LSTMSelfAttentionNetwork(input_size,hidden_size,output_size,num_layers, p_dropout).double().to(device)
trainableLayer = [True, True, False]
newModel = transferLSTM(model,newModel,trainableLayer)


muR = 3.96800e14
DU = 6378.1e3 # radius of earth in km
TU = ((DU)**3 / muR)**0.5

# initial conditions from example 2.7 in curtis
rLEO = np.array([6778e3,0])
vLEO = np.array([0, 8.435e3])
a = 8578e3
TLEO = 2*np.pi * np.sqrt(np.linalg.norm(a)**3 / muR)

e = 0.2098

mu = 1
a = a/DU
r = rLEO/DU
v = vLEO * TU / DU
T = TLEO / TU

J2 = 1.08263e-3

pam = [mu,J2]
h = np.cross(r,v)

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


n_epochs = 50
lr = 0.001
input_size = degreesOfFreedom
output_size = degreesOfFreedom
num_layers = 1
hidden_size = 50
p_dropout = 0.0
lookback = 1
p_motion_knowledge = 0.2


sysfuncptr = twoBodyPert
# sim time
t0, tf = 0, 5 * T

t = np.arange(t0, tf, TIME_STEP)
t = np.linspace(t0,tf,5000)
# initilize the arrays used to store the info from the numerical solution
IC = np.zeros((degreesOfFreedom,DATA_SET_SIZE))
output_seq = np.zeros((len(t),degreesOfFreedom))

# generate random data set of input thetas and output thetas and theta dots over a time series
IC = np.concatenate((r,v))

# numericResult = myRK4Py(sysfuncptr,IC,t,paramaters=pam)
_ , numericResult = ode45(sysfuncptr,[t0,tf],IC,t)

output_seq = numericResult

pertNR = numericResult

train_size = int(len(output_seq) * p_motion_knowledge)
test_size = len(output_seq) - train_size

train, test = output_seq[:train_size], output_seq[train_size:]

train_in,train_out = create_dataset(train,device,lookback=lookback)
test_in,test_out = create_dataset(test,device,lookback=lookback)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

optimizer = torch.optim.Adam(newModel.parameters(),lr=lr)

for epoch in range(n_epochs):

    trajPredition = plotPredition(epoch,newModel,'target',t=t*TU,output_seq=pertNR)

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

    print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))


circNR = nonDim2Dim4(circNR)
pertNR = nonDim2Dim4(pertNR)

plotOrbitPhasePredictions(circNR,'circular')
plotOrbitPhasePredictions(pertNR,'perturbed')
plt.show()

err = nonDim2Dim4(err)
plotPredition(epoch+1,newModel,'target',err,t*TU,output_seq=pertNR)
