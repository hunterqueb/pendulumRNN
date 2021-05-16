import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import math

L = 0.5
g = 9.81

b = 0.1
m = 1


def pendulumODE(theta,t):
    dtheta1 = theta[1]
    dtheta2 = -g/L*math.sin(theta[0])
    return [dtheta1, dtheta2]

def pendulumODEFriction(theta,t):
    dtheta1 = theta[1]
    dtheta2 = -b/m*theta[1]-g/L*math.sin(theta[0])
    return [dtheta1, dtheta2]

t0,tf = 0,10
t = np.arange(t0, tf, 0.01)

theta0 = [(math.pi/180) * 80, (math.pi/180) * 0]

r = integrate.odeint(pendulumODEFriction, theta0, t)
# LSODA is the closest integrator to ODE45 in matlab

# print(r.y.shape)
# print(r.y)

plt.plot(t,r[:,0])
plt.xlabel("Time (sec)")
plt.ylabel("Angluar Position (theta)")
plt.title("Numerical Solution for a Pendulum with Friction")
plt.show()
