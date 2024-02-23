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
from qutils.orbital import nonDim2Dim4, classicOrbitProp, ECI2OE

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

problemDim = 4 

mu = 396800
DU = 6378.1 # radius of earth in km
TU = ((DU)**3 / mu)**0.5

r_ijk_HEO = np.array((-1.26334513589522e-09,-3079387.27622166,-6149394.29413503))
v_ijk_HEO = np.array((10045.1938600227,-8.26237386395415e-13,-1.64995793440282e-12))

r_ijk_HEO = r_ijk_HEO/1000
v_ijk_HEO = v_ijk_HEO/1000
THEO = 11.9616 * 3600
OE = ECI2OE(r_ijk_HEO,v_ijk_HEO)

numPeriods = 5
nStepsPerOrbit = 1000

# non dimensionalize the OE
T = THEO / TU
OE[0] = OE[0] / DU
mu = 1

t = np.linspace(0,T*numPeriods,nStepsPerOrbit*numPeriods)

T_Class, X_Class = classicOrbitProp(t,OE,mu)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X_Class[:,0],X_Class[:,1],X_Class[:,2])
plt.show()

# from qutils.dynSys.dim6 import lagrangePlanEq

# # np.array((a, e, i, Omega, omega, M0, P))
# # elements = [OMEGA i omega a e M0]

# OE_lagrange = [OE[3],OE[2],OE[4],OE[0],OE[1],OE[5]]
# t_lagrange, x_lagrange = ode45(lagrangePlanEq,[0,T],OE_lagrange)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x_lagrange[:,0],x_lagrange[:,1],x_lagrange[:,2])
# plt.show()
