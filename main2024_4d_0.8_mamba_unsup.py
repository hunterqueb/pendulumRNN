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

from qutils.mamba import Mamba, MambaConfig

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


problemDim = 4

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

numPeriods = 20

n_epochs = 50
lr = 0.001
input_size = problemDim
output_size = problemDim
num_layers = 2
p_dropout = 0.0
lookback = 1
p_motion_knowledge = 1/numPeriods


config = MambaConfig(d_model=problemDim, n_layers=num_layers)
model = Mamba(config).to(device).double()

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = F.smooth_l1_loss

IC = torch.tensor(IC).to(device)

def trainUnsupervised():
    for epoch in range(n_epochs):

        # trajPredition = plotPredition(epoch,newModel,'target',t=t*TU,output_seq=pertNR)

        model.train()

        # give state to model
        y_pred = model(IC)
        
        # calculate loss from complex loss function
        loss = criterion(y_pred, twoBodyPert(0,(IC + y_pred * 0.001).cpu()))

        # back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # # Validation
        # model.eval()
        # with torch.no_grad():
        #     y_pred_train = model(train_in)
        #     train_loss = np.sqrt(criterion(y_pred_train, train_out).cpu())
        #     y_pred_test = model(test_in)
        #     test_loss = np.sqrt(criterion(y_pred_test, test_out).cpu())

        #     decAcc, err1 = findDecAcc(train_out,y_pred_train,printOut=False)
        #     decAcc, err2 = findDecAcc(test_out,y_pred_test)
        #     err = np.concatenate((err1,err2),axis=0)

        print("Epoch %d: train loss %.4f" % (epoch, loss))

trainUnsupervised()