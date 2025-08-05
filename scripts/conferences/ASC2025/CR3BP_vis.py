import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo
import sys

from qutils.integrators import ode87
from qutils.plot import plotCR3BPPhasePredictions,plotOrbitPredictions,plotStatePredictions, plot3dCR3BPPredictions,newPlotSolutionErrors, plotEnergy
from qutils.ml.utils import getDevice, printModelParmSize
from qutils.ml.regression import create_datasets, genPlotPrediction,transferMamba,LSTMSelfAttentionNetwork,transferLSTM, transferModelAll,LSTM, trainModel
from qutils.ml.superweight import printoutMaxLayerWeight,plotSuperWeight,plotMinWeight
from qutils.orbital import returnCR3BPIC, nonDim2Dim6, orbitInitialConditions, jacobiConstant6,dim2NonDim6
from qutils.tictoc import timer
from qutils.ml.mamba import Mamba, MambaConfig

# from mpldock import persist_layout
# plt.switch_backend('module://mpldock')
# plt.switch_backend('WebAgg')
# persist_layout('test')


compareLSTM = True
plotGen = True
plotOn = False

# get the orbit families from the command line arguments or if there is only 1 argument, select family pairs from 1 of 4 options
if len(sys.argv) > 2:
    sourceOrbitFamily = sys.argv[1]
    targetOrbitFamily = sys.argv[2]
elif len(sys.argv) == 2:
    # make sure arg is an int
    if not sys.argv[1].isdigit():
        raise ValueError("Argument must be an integer when no orbit family pairs are provided.")
    orbitPair = int(sys.argv[1])
    if orbitPair == 1:
        sourceOrbitFamily = 'longPeriod'
        targetOrbitFamily = 'shortPeriod'
        problemString = "LP-SP"
    elif orbitPair == 2:
        sourceOrbitFamily = 'longPeriod'
        targetOrbitFamily = 'dragonflySouth'
        problemString = "LP-DS"
    elif orbitPair == 3:
        sourceOrbitFamily = 'butterflyNorth'
        targetOrbitFamily = 'shortPeriod'
        problemString = "BN-SP"
    # elif orbitPair == 4:
    #     sourceOrbitFamily = 'halo'
    #     targetOrbitFamily = 'quasiperiodic'
    #     problemString = "H-QP"
    elif orbitPair == 5:
        sourceOrbitFamily = 'shortPeriod'
        targetOrbitFamily = 'butterflyNorth'
        problemString = "SP-BN"

else: # targeted orbit family pairs for transfer learning demonstration
    sourceOrbitFamily = 'longPeriod'
    targetOrbitFamily = 'shortPeriod'
    problemString = "LP-SP"

    sourceOrbitFamily = 'longPeriod'
    targetOrbitFamily = 'dragonflySouth'
    problemString = "LP-DS"

    # sourceOrbitFamily = 'butterflyNorth'
    # targetOrbitFamily = 'shortPeriod'
    # problemString = "BN-SP"

    # not using
    # sourceOrbitFamily = 'shortPeriod'
    # targetOrbitFamily = 'butterflyNorth'
    # problemString = "SP-BN"

    # sourceOrbitFamily = 'halo'
    # targetOrbitFamily = 'quasiperiodic'  # Note: 'quasiperiodic' is not a valid orbit family in JPL database, will use custom IC
    # problemString = "H-QP"

problemDim = 6
m_1 = 5.974E24  # kg
m_2 = 7.348E22 # kg
mu = m_2/(m_1 + m_2)

if sourceOrbitFamily == 'longPeriod':
    CR3BPIC_source = returnCR3BPIC(sourceOrbitFamily,L=4,id=751,stable=True)
elif sourceOrbitFamily == 'shortPeriod':
    CR3BPIC_source = returnCR3BPIC(sourceOrbitFamily,L=4,id=755,stable=True)
elif sourceOrbitFamily == 'halo':
    CR3BPIC_source = returnCR3BPIC(sourceOrbitFamily,id=928,L=1)
elif sourceOrbitFamily == 'butterflyNorth':
    CR3BPIC_source = returnCR3BPIC("butterfly",L="north",id=1080)
elif sourceOrbitFamily == 'butterflySouth':
    CR3BPIC_source = returnCR3BPIC("butterfly",L="south",id=270)
elif sourceOrbitFamily == 'dragonflyNorth':
    CR3BPIC_source = returnCR3BPIC("butterfly",L="north",id=404)
elif sourceOrbitFamily == 'dragonflySouth':
    CR3BPIC_source = returnCR3BPIC("butterfly",L="south",id=71)
else:
    raise ValueError("Invalid source orbit family.")


if targetOrbitFamily == 'shortPeriod':
    CR3BPIC_target = returnCR3BPIC(targetOrbitFamily,L=4,id=755,stable=True)
elif targetOrbitFamily == 'longPeriod':
    CR3BPIC_target = returnCR3BPIC(targetOrbitFamily,L=4,id=751,stable=True)
elif targetOrbitFamily == 'halo':
    CR3BPIC_target = returnCR3BPIC(targetOrbitFamily,id=928,L=1)
elif targetOrbitFamily == 'butterflyNorth':
    CR3BPIC_target = returnCR3BPIC("butterfly",L="north",id=1080)
elif targetOrbitFamily == 'butterflySouth':
    CR3BPIC_target = returnCR3BPIC("butterfly",L="south",id=270)
elif targetOrbitFamily == 'dragonflyNorth':
    CR3BPIC_target = returnCR3BPIC("butterfly",L="north",id=404)
elif targetOrbitFamily == 'dragonflySouth':
    CR3BPIC_target = returnCR3BPIC("butterfly",L="south",id=71)
elif targetOrbitFamily == 'quasiperiodic':
    CR3BPIC_target = orbitInitialConditions() # TODO - add quasiperiodic orbit ICs
else:
    raise ValueError("Invalid target orbit family.")


DU = 389703
TU = 382981
G = 6.67430e-11


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


device = getDevice()
modelString = 'Mamba'

if compareLSTM == True:
    numRuns = 2 # run the LSTM and Mamba models for comparison
else:
    numRuns = 1 # just run the Mamba model

for i in range(numRuns):
    x_0,tEnd = CR3BPIC_source()

    IC = np.array(x_0)

    numPeriods = 1

    t0 = 0; tf = numPeriods * tEnd

    delT = 0.001
    nSamples = int(np.ceil((tf - t0) / delT))
    t = np.linspace(t0, tf, nSamples)

    ODEtime = timer()
    t , numericResult = ode87(system,[t0,tf],IC,t,rtol=1e-9,atol=1e-12)
    ODEtime.toc()

    output_seq = numericResult
    output_seq_source = output_seq
    t_source = t / tEnd

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(output_seq[:, 0], output_seq[:, 1], output_seq[:, 2])

    # earthLocation = -mu
    # moonLocation = (1 - mu)

    # if DU:
    #     earthLocation = earthLocation * DU
    #     moonLocation = moonLocation * DU
    #     L = [l * DU for l in L]

    # if earth:
    #     ax.plot(earthLocation, 0, 0, 'ko', label='Earth')
    # if moon:
    #     ax.plot(moonLocation, 0, 0, 'go', label='Moon')

    # if sum(L) != 0:
    #     ax.plot([L[0]], [L[1]], [L[2]], 'd', color='grey', label='Lagrange Point')

    ax.set_title('Source Domain')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.zaxis.line.set_visible(False)
    ax.set_zticks([])
    ax.set_zlabel('')
    plt.rcParams['font.size'] = 10
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1))
    ax.legend()
    ax.grid(True)
    

    removeLast = 1000


    x_0,tEnd = CR3BPIC_target()

    IC = np.array(x_0)

    t0 = 0; tf = numPeriods * tEnd

    delT = 0.001
    nSamples = int(np.ceil((tf - t0) / delT))
    t = np.linspace(t0, tf, nSamples)

    t , numericResult = ode87(system,[t0,tf],IC,t,rtol=1e-15,atol=1e-15)

    output_seq_target = numericResult
    t_target = t / tEnd


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(output_seq_target[:-removeLast, 0], output_seq_target[:-removeLast, 1], output_seq_target[:-removeLast, 2])
    
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    zmin, zmax = ax.get_zlim()

    ax.set_title('Train Data')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.zaxis.line.set_visible(False)
    plt.rcParams['font.size'] = 10
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1))
    ax.legend()
    ax.grid(True)
    ax.set_yticks([])
    ax.set_ylabel('')



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(output_seq_target[:, 0], output_seq_target[:, 1], output_seq_target[:, 2],color='C1',label='Prediction Region')
    ax.plot(output_seq_target[:-removeLast, 0], output_seq_target[:-removeLast, 1], output_seq_target[:-removeLast, 2],color='C0',label='Train Data')
    ax.set_title('Target Domain')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.zaxis.line.set_visible(False)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    plt.rcParams['font.size'] = 10
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1))
    ax.legend()
    ax.grid(True)
    ax.set_yticks([])
    ax.set_ylabel('')


    plt.show()
