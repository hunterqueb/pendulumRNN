
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import random
import torch.nn.functional as F
import torch.utils.data as data
import scipy as sp

from qutils.integrators import myRK4Py, ode45
from qutils.ml import Adam_mini,printModelParameters, getDevice, create_datasets
from qutils.mlExtras import findDecAcc, generateTrajectoryPrediction
from qutils.plot import plotOrbitPhasePredictions, plotStatePredictions, plotSolutionErrors
from qutils.orbital import nonDim2Dim4

from qutils.ml.regression import create_datasets as create_dataset

from qutils.ml.mamba import Mamba, MambaConfig

# seed any random functions
random.seed(123)
device = getDevice()


plot = True


problemDim = 4 

TIME_STEP = 0.05

# transfer to different system

# newModel = LSTMSelfAttentionNetwork(input_size,hidden_size,output_size,num_layers, p_dropout).double().to(device)
# trainableLayer = [True, True, False]
# newModel = transferLSTM(model,newModel,trainableLayer)


muR = 396800
DU = 6378.1 # radius of earth in km
TU = ((DU)**3 / muR)**0.5

p = 20410 # km
e = 0.8

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
linPam = [mu,IC]

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

    # a_drag_x = 0
    # a_drag_y = 0
    # j2_accel_x = 0
    # j2_accel_y = 0
    dydt3 = -mu / R**3 * y[0] + j2_accel_x + a_drag_x
    dydt4 = -mu / R**3 * y[1] + j2_accel_y + a_drag_y

    return np.array([dydt1, dydt2,dydt3,dydt4])


def twoBodyLinearized(t, y, p=linPam):
    r = y[0:2]
    R = np.linalg.norm(r)
    IC = p[1]
    r0 = np.linalg.norm(IC[0:2])

    mu = p[0]
    dydt1 = y[2]
    dydt2 = y[3]

    dydt3 = -mu / r0**3 * y[0]
    dydt4 = -mu / r0**3 * y[1]

    return np.array([dydt1, dydt2,dydt3,dydt4])


n_epochs = 5
lr = 0.001
input_size = problemDim
output_size = problemDim
num_layers = 1
p_dropout = 0.0
lookback = 1
p_motion_knowledge = 1/2

sysfuncptr = twoBodyPert
# sim time

t0, tf = 0, 1 * T

t_range = np.arange(t0, tf, TIME_STEP)

IC = np.concatenate((r,v))

t , numericResult = ode45(sysfuncptr,[t0,tf],IC,t_range)
t_sol , numericResult_sol = ode45(twoBodyPert,[t0,tf],IC,t_range)

output_seq = numericResult

pertNR = numericResult

numericResult_sol_nonDim = numericResult_sol

config = MambaConfig(d_model=problemDim, n_layers=1,d_state=problemDim,expand_factor=1)
newModel = Mamba(config).to(device).double()
optimizer = Adam_mini(newModel,lr=lr)
criterion = F.smooth_l1_loss


train_size = int(len(pertNR) * p_motion_knowledge)
train_size = 2
test_size = len(pertNR) - train_size

train_in,train_out,test_in,test_out = create_datasets(pertNR,1,train_size,device)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

for epoch in range(n_epochs):

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

if plot:
    trajPredition = plotStatePredictions(newModel,t,numericResult_sol,train_in,test_in,train_size,test_size,units=("DU","DU","DU/TU","DU/TU"))
    trajPredition_norm = trajPredition
    trajPredition = nonDim2Dim4(trajPredition)

pertNR = nonDim2Dim4(pertNR)
output_seq = nonDim2Dim4(output_seq)
numericResult_sol = nonDim2Dim4(numericResult_sol)

printModelParameters(newModel)

if plot:
    plotOrbitPhasePredictions(output_seq,trajPredition,earth=None)
    plt.plot(numericResult_sol[:, 0], numericResult_sol[:, 1], label='Nonlinear Solution')
    plt.legend()

    plotSolutionErrors(output_seq,trajPredition,t)


A_mamba_t = newModel.layers[0].mixer.A_SSM
B_mamba_t = newModel.layers[0].mixer.B_SSM
C_mamba_t = newModel.layers[0].mixer.C_SSM
D_mamba_t = newModel.layers[0].mixer.D_SSM
delta_mamba = newModel.layers[0].mixer.delta

A_mamba=newModel.layers[0].mixer.A_SSM.cpu().numpy()
B_mamba = newModel.layers[0].mixer.B_SSM[:,-1,:].cpu().numpy()
C_mamba = newModel.layers[0].mixer.C_SSM[:,-1,:].cpu().numpy()
D_mamba = newModel.layers[0].mixer.D_SSM.cpu().numpy()

eigVals,eigVects = np.linalg.eig(A_mamba)
print(newModel.layers[0].mixer.B_SSM.shape)

print('rank of (A - eig*I) = %i with n = %i' % (np.linalg.matrix_rank(np.concatenate((A_mamba - (np.eye(problemDim) * eigVals), B_mamba), axis=0)),problemDim))

# dr_mamba = sp.linalg.expm(A_mamba) @ IC.reshape((problemDim,1))

# print(C_mamba @ dr_mamba)

y = newModel.layers[0].mixer.selective_scan(train_in,delta_mamba,A_mamba_t,B_mamba_t,C_mamba_t,D_mamba_t).cpu().numpy()


dr = sp.linalg.expm(np.asarray(((0,0,1,0),(0,0,0,1),(-mu/np.linalg.norm(r)**3,0,0,0),(0,-mu/np.linalg.norm(r)**3,0,0))) * t[1]) @ IC.reshape((problemDim,1))


nondim_initial = IC
nondim_stm = dr.T.reshape(problemDim)
nondim_ssm = y[0,0,:]
nondim_network = trajPredition_norm[1,:]
nondim_nonlinear = numericResult_sol_nonDim[1,:]



dim_initial = nonDim2Dim4(IC.reshape(1,problemDim)).reshape(problemDim)
dim_stm = nonDim2Dim4(dr.T[0].reshape(1,problemDim)).reshape(problemDim)
dim_ssm = nonDim2Dim4(y[0,0,:].reshape(1,problemDim)).reshape(problemDim)
dim_network = nonDim2Dim4(trajPredition_norm[1,:].reshape(1,problemDim)).reshape(problemDim)
dim_nonlinear = numericResult_sol[1,:]

print('\nIn Nondimensional')
print("Initial Conditions",nondim_initial)
print("STM Solution at t = 0.05 TU",nondim_stm)
print("SSM Output at t = 0.05 TU",nondim_ssm)
print("Network Prediction provided by full mamba at t = 0.05 TU: ",nondim_network)
print("Nonlinear Solution at t = 0.05 TU: ",nondim_nonlinear)

print('\nIn Dimensional')
print("Initial Conditions",dim_initial)
print("STM Solution at t = 0.05 TU",dim_stm)
print("SSM Output at t = 0.05 TU",dim_ssm)
print("Network Prediction provided by full mamba at t = 0.05 TU: ",dim_network)
print("Nonlinear Solution at t = 0.05 TU: ",dim_nonlinear)


nondim_ssm_mamba_diff = nondim_network - nondim_ssm
dim_ssm_mamba_diff = dim_network - dim_ssm

nondim_mamba_nonlin_diff = nondim_network - nondim_nonlinear
dim_mamba_nonlin_diff = dim_network - dim_nonlinear

nondim_ssm_nonlin_diff = nondim_ssm - nondim_nonlinear
dim_ssm_nonlin_diff = dim_ssm - dim_nonlinear


nondim_stm_nonlinear_diff = nondim_stm - nondim_nonlinear
dim_stm_nonlinear_diff = dim_stm - dim_nonlinear

print('\nNondim difference btwn ssm & mamba',nondim_ssm_mamba_diff)
print('Dim difference btwn ssm & mamba',dim_ssm_mamba_diff)

print('\nNondim difference btwn mamba & nonlinear soln',nondim_mamba_nonlin_diff)
print('Dim difference btwn mamba & nonlinear soln',dim_mamba_nonlin_diff)

print('\nNondim difference btwn ssm & nonlinear soln',nondim_ssm_nonlin_diff)
print('Dim difference btwn ssm & nonlinear soln',dim_ssm_nonlin_diff)

print('\nNondim Difference btwn STM & nonlinear soln',nondim_stm_nonlinear_diff)
print('Dim Difference btwn STM & nonlinear soln',dim_stm_nonlinear_diff)
if plot:
    plt.show()
