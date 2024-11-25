import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import random
import torch.nn.functional as F
import torch.utils.data as data

from qutils.integrators import myRK4Py, ode45
from qutils.mlExtras import findDecAcc
from qutils.plot import plotOrbitPhasePredictions, plotSolutionErrors
from qutils.orbital import nonDim2Dim4
from qutils.ml import getDevice
from qutils.mamba import Mamba, MambaConfig

from nets import create_dataset
device = getDevice()

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
t0, tf = 0, T
dt = 0.01
t = np.arange(t0, tf, dt)

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
lookback = 1
p_motion_knowledge = 1/numPeriods


config = MambaConfig(d_model=problemDim, n_layers=num_layers)
model = Mamba(config).to(device).double()

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = F.smooth_l1_loss

train_in = torch.tensor(IC.reshape(1,1,4)).to(device)
t_output_seq , output_seq = ode45(twoBodyPert,(t0,tf),IC,t)


def trainUnsupervised():
    for epoch in range(n_epochs):

        model.train()

        # give state to model
        y_pred = model(train_in)
        
        # calculate loss from complex loss function
        sysOut = torch.tensor(twoBodyPert(0,y_pred.detach().cpu().numpy().reshape(problemDim)).reshape(1,1,problemDim)).to(device)
        loss = criterion((train_in - y_pred)/dt, sysOut)

        # back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch %d: train loss %.4f" % (epoch, loss))

trainUnsupervised()

data_out_array = np.empty((1, int(tf/dt)+1, problemDim))
def generateTrajectory():
    print(len(t))
    data_in = train_in
    model.eval()
    with torch.no_grad():
        for i in range(len(t)):
            data_out = model(data_in)
            data_out_array[0,i,:] = data_out.cpu().numpy()
            data_in = data_out
            # print(i)
    y_prediction = np.squeeze(data_out_array)
    return y_prediction

y_prediction = generateTrajectory()

plotSolutionErrors(output_seq,y_prediction,t)
plt.figure()
plotOrbitPhasePredictions(output_seq)
plotOrbitPhasePredictions(y_prediction)
plt.show()