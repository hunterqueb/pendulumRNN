import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data

from qutils.integrators import ode45
from qutils.integrators import ode1412
import desolver as de
from memory_profiler import profile
plotOn = True

problemDim = 4 

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
initialConditions = np.radians(np.random.uniform(-180, 180, (problemDim,)))

tStart = 0
tEnd = 20
tSpan = np.array([tStart,tEnd])
dt = 0.01
tSpanExplicit = np.linspace(tStart,tEnd,int(tEnd / dt))

t, numericResult = ode1412(doublePendulumODE,tSpan,initialConditions,tSpanExplicit)

if plotOn is True:
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(numericResult[:,0],numericResult[:,2],'r',label = "Truth")
    plt.xlabel('Theta 1')
    plt.ylabel('Theta 1 Dot')
    plt.axis('equal')


    plt.subplot(2, 1, 2)
    plt.plot(numericResult[:,1],numericResult[:,3],'r',label = "Truth")
    plt.xlabel('Theta 2')
    plt.ylabel('Theta 2 Dot')
    plt.axis('equal')
    plt.legend()

    plt.show()


