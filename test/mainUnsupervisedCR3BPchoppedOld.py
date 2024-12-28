import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

from qutils.ml import getDevice
import qutils.trialSolution as TS 
from qutils.mlExtras import findDecAcc as findDecimalAccuracy
from qutils.tictoc import timer
from qutils.pinn import trainForwardLagAna,FeedforwardSin,FeedforwardCos
from qutils.integrators import ode87
from qutils.plot import plotCR3BPPhasePredictions,plotOrbitPredictions, plotSolutionErrors, plotEnergy
from qutils.orbital import jacobiConstant

arg1 = None
if sys.argv[1:]:   # test if there are atleast 1 argument
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    plotOff = int(arg1)
    nTest = int(arg2)
else:
    plotOff = 0
    nTest = 100

DEBUG = False

problemDim = 4 
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
# x_0 = 3.0680940168765748E-2	
# y_0 = 0
# vx_0 = -1.9798683479304902E-12	
# vy_0 = 6.6762053214726942E+0	
# tEnd = 6.3044730125061266E+0	

# # short period l4 liberation point id 1062
# x_0 = 4.8784941344943100E-1	
# y_0 = 1.0231615681834254E+0	
# vx_0 = 2.4100203799532632E-1	
# vy_0 = -1.0386356858552348E-1	
# tEnd = 6.5699439082821405

# # Low Prograde Orbit id 1336 "moon orbit"
# x_0 = 1.0676492111091758E+0
# y_0 = 0.0000000000000000E+0
# vx_0 = 6.1042623464139881E-15
# vy_0 = 3.0241486917222471E-1
# tEnd = 1.3463994791934020E+0	

# # Long Period L4 id 123 Stability: 6.1992765524940317E+1
# x_0 = 4.8784941344943100E-1
# y_0 = 3.5889770411166416E-1
# vx_0 = -8.5012625872098058E-1
# vy_0 = 3.0599104949396405E-1
# tEnd = 2.5791531978360762E+1	

# # Long Period L4 id 1173
# x_0 = 4.8784941344943100E-1
# y_0 = 6.7241857567449559E-1
# vx_0 = -2.4475661452281838E-1
# vy_0 = 1.3812226057131133E-1
# tEnd = 2.4047875532062960E+1

# # Lyapunov Orbit Family around L2 id 972: stability - 5.9998680574258844E+2
# x_0 =  1.1192937490150918E+0
# y_0 =  0.0000000000000000E+0
# vx_0 = 2.6418584656601576E-15
# vy_0 = 1.8122474967618102E-1
# tEnd = 3.4181096055680626E+0	

# # Lyapunov Orbit Family around L2 id 540 : stability - 5.2052887366713769E+1
# x_0 = 9.9963706922393214E-1
# y_0 = 0.0000000000000000E+0
# vx_0 = -1.1109531673642259E-14
# vy_0 = 1.4398178227943141E+0
# tEnd = 6.2560355541392445E+0	

vSquared = (vx_0**2 + vy_0**2)
xn1 = -mu
xn2 = 1-mu
rho1 = np.sqrt((x_0-xn1)**2+y_0**2)
rho2 = np.sqrt((x_0-xn2)**2+y_0**2)

C0 = (x_0**2 + y_0**2) + 2*(1-mu)/rho1 + 2*mu/rho2 - vSquared
print('Jacobi Constant: {}'.format(C0))

# mu = 0.012277471
# x_0 = 0.994
# y_0 = 0
# vx_0 = 0
# vy_0 = -2.0317326295573368357302057924


# Then stack everything together into the state vector
r_0 = np.array((x_0, y_0))
v_0 = np.array((vx_0, vy_0))
x_0 = np.hstack((r_0, v_0))

netptr = FeedforwardSin
netptrd = FeedforwardCos

def system(t, Y,mu=mu):
    """Solve the CR3BP in nondimensional coordinates.
    
    The state vector is Y, with the first three components as the
    position of $m$, and the second three components its velocity.
    
    The solution is parameterized on $\\pi_2$, the mass ratio.
    """
    # Get the position and velocity from the solution vector
    x, y = Y[:2]
    xdot, ydot = Y[2:]

    # Define the derivative vector

    dydt1 = xdot
    dydt2 = ydot
    sigma = np.sqrt(np.sum(np.square([x + mu, y])))
    psi = np.sqrt(np.sum(np.square([x - 1 + mu, y])))
    dydt3 = 2 * ydot + x - (1 - mu) * (x + mu) / sigma**3 - mu * (x - 1 + mu) / psi**3
    dydt4 = -2 * xdot + y - (1 - mu) * y / sigma**3 - mu * y / psi**3
    return np.array([dydt1, dydt2,dydt3,dydt4])

def systemTensor(t, Y, mu=mu):
    # Get the position and velocity from the solution vector
    x = Y[:,0].reshape(-1,1)
    y = Y[:,1].reshape(-1,1)
    xdot = Y[:,2].reshape(-1,1)
    ydot = Y[:,3].reshape(-1,1)

    # Define the derivative vector
    Ydot1 = xdot
    Ydot2 = ydot

    sigma = torch.sqrt(torch.sum(torch.square(torch.cat([x + mu, y],dim=1)),dim=1)).reshape(-1,1)
    psi = torch.sqrt(torch.sum(torch.square(torch.cat([x - 1 + mu, y],dim=1)),dim=1)).reshape(-1,1)
    Ydot3 = 2 * ydot + x - (1 - mu) * (x + mu) / sigma**3 - mu * (x - 1 + mu) / psi**3
    Ydot4 = -2 * xdot + y - (1 - mu) * y / sigma**3 - mu * y / psi**3
    return torch.cat((Ydot1,Ydot2,Ydot3,Ydot4),1)


IC = x_0
y0 = np.array(IC)

device = getDevice()

y0Tensor = torch.from_numpy(y0).to(device)

dataSet = 10 * 50
epochs = 100
learningRate = 6e-4

tStart = 0
delT = 0.001
nSamples = int(np.ceil((tEnd - tStart) / delT))
T = tEnd
t = np.linspace(tStart, tEnd, nSamples)
# manually configure points of time 
tParts = [tStart, T/4, T/2, 3*T/4, tEnd]

# or automatically set them
desiredSegs = 4

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
debugList = [None for _ in range(numSegs)]

inputSize = 1
hiddenLayerSize = 10
outputSize = 1

nets = [[0 for _ in range(problemDim)] for _ in range(numSegs)]
optimizers = [[0 for _ in range(problemDim)] for _ in range(numSegs)]
trialSolutions = [0 for _ in range(numSegs)]
trainingDataInputs = [0 for _ in range(numSegs)]

t = []
T = []

yTruth = np.empty((0, problemDim))
yTruthSeg = [0 for _ in range(numSegs)]
yTest = np.empty((0, problemDim))
y0Seg = y0

for i in range(numSegs):
    for j in range(problemDim):
        # check which network you are constructing in the dimension to keep kinematic consistency
        if j < problemDim/2:
            net = netptr(inputSize, hiddenLayerSize,outputSize).double()
        else:
            net = netptrd(inputSize, hiddenLayerSize,outputSize).double()
        
        net.to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=learningRateList[i])

        nets[i][j] = net
        optimizers[i][j] = optimizer
    
    trainingDataInputs[i] = np.reshape(np.linspace(tParts[i], tParts[i+1], dataSet),[dataSet,1])
    trainingDataInputs[i] = torch.from_numpy(trainingDataInputs[i]).double().requires_grad_(True)  # 2000, 4

    trainingDataInputs[i] = trainingDataInputs[i].to(device)

    trialSolutions[i] = TS.Sin(y0Seg,torch.from_numpy(y0Seg).to(device).reshape(1,-1),tParts[i], tParts[i+1],jacobiConst = C0)

    tSeg = np.linspace(tParts[i], tParts[i+1], nSamples)
    
    tSeg, yTruthSeg[i] = ode87(system, (tParts[i], tParts[i+1]), y0Seg, rtol=1e-8,atol=1e-10,t_eval=tSeg)
    
    T.append(torch.tensor(np.reshape(tSeg, [len(tSeg), 1])).to(device))

    trialSolutions[i].setTime(T[i], tSeg)
    trialSolutions[i].setTruth(yTruthSeg[i])
    trialSolutions[i].setPlot(plotCR3BPPhasePredictions)
    trialSolutions[i].setKC(True)
    
    y0Seg = yTruthSeg[i][-1,:]

    yTruth = np.append((yTruth), (yTruthSeg[i]), axis=0)
    t = np.append((t),(tSeg))
    

criterion1 = torch.nn.MSELoss()
# criterion1 = torch.nn.HuberLoss()


# set to train
for segs in nets:
    for element in segs:
        element.train()

# train each set of networks
timeToTrain = timer()
for i in range(numSegs):
    print('\nTraining Segment {}'.format(i+1))
    trainForwardLagAna(nets[i], epochList[i], optimizers[i], criterion1, trainingDataInputs[i], systemTensor, trialSolutions[i],debug=debugList[i])

print('Total Time to Train: {}'.format(timeToTrain.tocVal()))

# set to evaluation
for segs in nets:
    for element in segs:
        element.eval()

# evaluate networks
for i in range(numSegs):
    y_predL = []
    for j in range(problemDim):
        y_pred = nets[i][j](T[i])
        y_predL.append(y_pred)
    y_pred = torch.cat(y_predL,1)
    yTestSeg = trialSolutions[i](y_pred, T[i]).cpu().squeeze().detach().numpy()
    yTest = np.append((yTest), (yTestSeg), axis=0)
    # print section accuracy
    print('\nSection {}'.format(i+1))
    findDecimalAccuracy(yTruthSeg[i], yTestSeg)

# full trajectory accuracy
print('\nFull Trajectory')
decAcc, avg = findDecimalAccuracy(yTruth, yTest)
print("Final State Error",(yTruth-yTest)[-1],' in normalized units')

netTime = timer()
netTime.tic()
tTest = torch.tensor(np.ones((nTest,1)) * tEnd,device=device)
for j in range(problemDim):
    y_pred = nets[-1][j](tTest)
    y_predL.append(y_pred)
yTestNN = trialSolutions[-1](y_pred, tTest).cpu().squeeze().detach().numpy()
finalNet = netTime.tocVal()
print('Time to evaluate {} ODES with NN: {}'.format(nTest,finalNet))

odeTime = timer()
odeTime.tic()
for i in range(nTest):
    tSeg, numericalResult  = ode87(system, (0, tEnd), y0Seg, rtol=1e-8,atol=1e-10)
finalOde = odeTime.tocVal()
print('Time to evaluate {} ODES with RK45: {}'.format(nTest,finalOde))

plotCR3BPPhasePredictions(yTruth,yTest)
plotOrbitPredictions(yTruth,yTest,t=t)
plotSolutionErrors(yTruth,yTest,t)
plotEnergy(yTruth,yTest,t,jacobiConstant,yLabel='Jacobi Constant')

if(not plotOff):
    plt.show()
