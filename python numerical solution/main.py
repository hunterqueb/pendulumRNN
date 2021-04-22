import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import math

L = 0.5
g = 9.81

b = 0.3
m = 1

def pendulumODE(t,theta):
    dtheta1 = theta[1]
    dtheta2 = -g/L*math.sin(theta[0])
    return dtheta1, dtheta2


def pendulumODEFriction(t, theta):
    dtheta1 = theta[1]
    dtheta2 = -b/m*theta[1]-g/L*math.sin(theta[0])
    return dtheta1, dtheta2

t0,tf = 0,20
t = np.linspace(t0, tf, 100)

theta0 = [(math.pi/180) * 80, (math.pi/180) * 0]

r = integrate.solve_ivp(pendulumODEFriction, (t0, tf), theta0, "LSODA")
# LSODA is the closest integrator to ODE45 in matlab

print(r.y.shape)
# print(r.y)


plt.plot(r.t,r.y[0])
plt.show()
