import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import random
import torch.nn.functional as F
import torch.utils.data as data

# from quebutils.integrators import myRK4Py, ode45
# from quebutils.mlExtras import findDecAcc

from quebUtils.integrators import myRK4Py, ode45
from quebUtils.mlExtras import findDecAcc
from quebUtils.plot import plotOrbitPhasePredictions
from quebUtils.orbital import OE2ECI

DU = 6378.1 # radius of earth in km
TU = 806.80415

muR = 396800

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

IC = np.concatenate((r,v))

print(rHEO)
print(vHEO)
print(IC)

def twoBodyCirc(t, y, p=mu):
    r = y[0:2]
    R = np.linalg.norm(r)

    dydt1 = y[2]
    dydt2 = y[3]

    dydt3 = -p / R**3 * y[0]
    dydt4 = -p / R**3 * y[1]

    return np.array([dydt1, dydt2,dydt3,dydt4])

t,y = ode45(twoBodyCirc,[0,T],IC)

plt.figure()
plt.plot(y[:,0],y[:,1])
plt.plot(0,0,'ko')
plt.show()