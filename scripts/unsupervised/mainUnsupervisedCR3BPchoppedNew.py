import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

from qutils.ml import getDevice
import qutils.trialSolution as TS 
from qutils.mlExtras import findDecAcc as findDecimalAccuracy
from qutils.tictoc import timer
from qutils.pinn import PINN,FeedforwardSin,FeedforwardCos
from qutils.integrators import ode87
from qutils.plot import plotCR3BPPhasePredictions,plotOrbitPredictions, plotSolutionErrors, plotEnergy, plot3dCR3BPPredictions
from qutils.orbital import  returnCR3BPIC,nonDim2Dim6

arg1 = None
if sys.argv[1:]:   # test if there are atleast 1 argument
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    plotOff = int(arg1)
    nTest = int(arg2)
else:
    plotOff = 0
    nTest = 100

DEBUG = True

problemDim = 6
m_1 = 5.974E24  # kg
m_2 = 7.348E22 # kg
mu = m_2/(m_1 + m_2)

# orbits selected from
# https://ssd.jpl.nasa.gov/tools/periodic_orbits.html

# short period L4 liberation point id 1 'large stable'
x_0 = 0.487849413
y_0 = 1.471265959
vx_0 = 1.024841387
vy_0 = -0.788224219
tEnd = 6.2858346244258847

# # long period L4 liberation point id 708 'around L4 only'
# x_0 = 4.8784941344943100E-1	
# y_0 = 7.9675359028611403E-1	
# vx_0 = -7.4430997318144260E-2	
# vy_0 = 5.6679773588495463E-2
# tEnd = 2.1134216469590449E1 * 4


# # Distant Retrograde Orbit id 1  - Stability: 1.0000656297257782E+0 

# # short period l4 liberation point id 1062

# # Low Prograde Orbit id 1336 "moon orbit"
# x_0 = 1.0676492111091758E+0
# y_0 = 0.0000000000000000E+0
# vx_0 = 6.1042623464139881E-15
# vy_0 = 3.0241486917222471E-1
# tEnd = 1.3463994791934020E+0	

# # Long Period L4 id 123 Stability: 6.1992765524940317E+1

# # Long Period L4 id 1173

# # Lyapunov Orbit Family around L2 id 972: stability - 5.9998680574258844E+2
# # Lyapunov Orbit Family around L2 id 540 : stability - 5.2052887366713769E+1


def jacobiConstant(Y):
    mu = 0.012277471  # Mass ratio (m_2 / (m_1 + m_2))
    x = Y[0]
    y = Y[1]
    z = Y[2]
    xdot = Y[3]
    ydot = Y[4]
    zdot = Y[5]
    
    vSquared = (xdot**2 + ydot**2 + zdot**2)
    
    xn1 = -mu
    xn2 = 1 - mu
    
    rho1 = np.sqrt((x - xn1)**2 + y**2 + z**2)
    rho2 = np.sqrt((x - xn2)**2 + y**2 + z**2)
    
    C = (x**2 + y**2 + z**2) + 2 * (1 - mu) / rho1 + 2 * mu / rho2 - vSquared
    
    return C

# Then stack everything together into the state vector

orbitFamily = 'halo'

CR3BPIC = returnCR3BPIC(orbitFamily,L=1,id=894,stable=True)

x_0,tEnd = CR3BPIC()

C0 = jacobiConstant(x_0)

netptr = FeedforwardSin
netptrd = FeedforwardCos

def system(t, Y,mu=mu):
    """Solve the CR3BP in nondimensional coordinates.
    
    The state vector is Y, with the first three components as the
    position of $m$, and the second three components its velocity.
    
    The solution is parameterized on $\\pi_2$, the mass ratio.
    """
    # Get the position and velocity from the solution vector
    x, y, z = Y[:3]
    xdot, ydot, zdot = Y[3:]

    # Define the derivative vector

    dydt1 = xdot
    dydt2 = ydot
    dydt3 = zdot

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)

    dydt4 = 2 * ydot + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    dydt5 = -2 * xdot + y - (1 - mu) * y / r1**3 - mu * y / r2**3
    dydt6 = -(1 - mu) * z / r1**3 - mu * z / r2**3

    return np.array([dydt1, dydt2,dydt3,dydt4,dydt5,dydt6])

def systemTensor(t, Y, mu=mu):
    # Get the position and velocity from the solution vector
    x = Y[:,0].reshape(-1,1)
    y = Y[:,1].reshape(-1,1)
    z = Y[:,2].reshape(-1,1)
    xdot = Y[:,3].reshape(-1,1)
    ydot = Y[:,4].reshape(-1,1)
    zdot = Y[:,5].reshape(-1,1)

    # Define the derivative vector
    Ydot1 = xdot
    Ydot2 = ydot
    Ydot3 = zdot

    r1 = torch.sqrt(torch.sum(torch.square(torch.cat([x + mu, y, z], dim=1)), dim=1)).reshape(-1, 1)
    r2 = torch.sqrt(torch.sum(torch.square(torch.cat([x - 1 + mu, y, z], dim=1)), dim=1)).reshape(-1, 1)

    sigma = torch.sqrt(torch.sum(torch.square(torch.cat([x + mu, y],dim=1)),dim=1)).reshape(-1,1)
    psi = torch.sqrt(torch.sum(torch.square(torch.cat([x - 1 + mu, y],dim=1)),dim=1)).reshape(-1,1)

    Ydot4 = 2 * ydot + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    Ydot5 = -2 * xdot + y - (1 - mu) * y / r1**3 - mu * y / r2**3
    Ydot6 = -(1 - mu) * z / r1**3 - mu * z / r2**3
    return torch.cat((Ydot1,Ydot2,Ydot3,Ydot4,Ydot5,Ydot6),1)


IC = x_0
y0 = np.array(IC)

device = getDevice()

y0Tensor = torch.from_numpy(y0).to(device)

dataSet = 10 * 50
epochs = 10000
learningRate = 6e-4

numPeriods = 5

tEnd = tEnd * numPeriods

tStart = 0
delT = 0.001
nSamples = int(np.ceil((tEnd - tStart) / delT))
T = tEnd
t = np.linspace(tStart, tEnd, nSamples)
# manually configure points of time 
tParts = [tStart, T/4, T/2, 3*T/4, tEnd]

# or automatically set them
desiredSegs = 20 * numPeriods

tParts = np.linspace(tStart,tEnd, desiredSegs + 1)

# ## EXPANDING THE TIME SERIES IN BAD PARTS OF THE APPROXIMATION

# lowerInd = 8
# higherInd = 15

# expandedSegs = (higherInd - lowerInd) * 2

# expandLower = tParts[lowerInd]
# expandHigher = tParts[higherInd]
# tPartsExpand = np.linspace(expandLower,expandHigher, expandedSegs + 1)
# tParts = np.concatenate((tParts[0:lowerInd],tPartsExpand,tParts[int(higherInd+1):]))


# for desiredSegs = 10
# expandLower = tParts[2]
# expandHigher = tParts[3]
# tPartsExpand = np.linspace(expandLower,expandHigher, expandedSegs + 1)
# tParts = np.concatenate((tParts[0:2],tPartsExpand,tParts[4:]))

numSegs = len(tParts) - 1

epochList = [epochs for _ in range(numSegs)]
learningRateList = [learningRate for _ in range(numSegs)]

inputSize = 1
hiddenLayerSize = 10
outputSize = 1

pinnSolution = PINN(problemDim,device,netptr,netptrd)
pinnSolution.setupNetwork(inputSize,hiddenLayerSize,outputSize,learningRateList,dataSet,tParts)
pinnSolution.setupTrialSolution(system,y0)
pinnSolution.setupConservation(jacobiConst=C0)
pinnSolution.setPlot(plotCR3BPPhasePredictions)
pinnSolution.setToTrain()

criterion = torch.nn.MSELoss()
# criterion1 = torch.nn.HuberLoss()

# train each set of networks
timeToTrain = timer()
pinnSolution.trainAnalytical(systemTensor,criterion,epochList)
print('Total Time to Train: {}'.format(timeToTrain.tocVal()))

# set to evaluation
pinnSolution.setToEvaluate()

# evaluate networks
t,yTest = pinnSolution()

t=t/tEnd

yTruth = pinnSolution.getTrueSolution()

# full trajectory accuracy
print('\nFull Trajectory')
decAcc, avg = findDecimalAccuracy(yTruth, yTest)
print("Final State Error",(yTruth-yTest)[-1],' in normalized units')


DU = 389703
TU = 382981


# netTime = timer()
# netTime.tic()
# tTest = torch.tensor(np.ones((nTest,1)) * tEnd,device=device)
# pinnSolution.testEvaulation(tTest)
# finalNet = netTime.tocVal()
# print('Time to evaluate {} ODES with NN: {}'.format(nTest,finalNet))

# odeTime = timer()
# odeTime.tic()
# for i in range(nTest):
#     tSeg, numericalResult  = ode87(system, (0, tEnd), y0, rtol=1e-8,atol=1e-10)
# finalOde = odeTime.tocVal()
# print('Time to evaluate {} ODES with RK45: {}'.format(nTest,finalOde))


plotCR3BPPhasePredictions(yTruth,yTest)
plot3dCR3BPPredictions(yTruth,yTest,L=1)

yTruth = nonDim2Dim6(yTruth,DU,TU)
yTest = nonDim2Dim6(yTest,DU,TU)

plotSolutionErrors(yTruth,yTest,t)

errorAvg = np.nanmean(abs(yTest-yTruth), axis=0)
print("Average values of each dimension:")
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg}")

if(not plotOff):
    plt.show()
