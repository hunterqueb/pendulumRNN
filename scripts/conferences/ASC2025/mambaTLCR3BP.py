import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo
import sys

from qutils.integrators import ode87
from qutils.plot import plotCR3BPPhasePredictions,plotOrbitPredictions,plotStatePredictions, plot3dCR3BPPredictions,newPlotSolutionErrors, plotEnergy
from qutils.ml import getDevice, create_datasets, genPlotPrediction,transferMamba,LSTMSelfAttentionNetwork,transferLSTM, transferModelAll,LSTM, trainModel
from qutils.mlExtras import findDecAcc, plotSuperWeight, plotMinWeight, printoutMaxLayerWeight
from qutils.orbital import returnCR3BPIC, nonDim2Dim6, orbitInitialConditions, jacobiConstant6,dim2NonDim6
from qutils.tictoc import timer
from qutils.mamba import Mamba, MambaConfig

# from mpldock import persist_layout
# plt.switch_backend('module://mpldock')
plt.switch_backend('WebAgg')
# persist_layout('test')


compareLSTM = True

# get the source and target orbit families from sys args
if len(sys.argv) > 1:
    sourceOrbitFamily = sys.argv[1]
    targetOrbitFamily = sys.argv[2]
else: # targeted orbit family pairs for transfer learning demonstration
    sourceOrbitFamily = 'longPeriod'
    targetOrbitFamily = 'shortPeriod'

    # sourceOrbitFamily = 'longPeriod'
    # targetOrbitFamily = 'dragonflySouth'

    # sourceOrbitFamily = 'butterflyNorth'
    # targetOrbitFamily = 'shortPeriod'

    # not using
    # sourceOrbitFamily = 'shortPeriod'
    # targetOrbitFamily = 'butterflyNorth'

    # sourceOrbitFamily = 'halo'
    # targetOrbitFamily = 'quasiperiodic'  # Note: 'quasiperiodic' is not a valid orbit family in JPL database, will use custom IC

plotOn = True

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

    numPeriods = 5

    t0 = 0; tf = numPeriods * tEnd

    delT = 0.001
    nSamples = int(np.ceil((tf - t0) / delT))
    t = np.linspace(t0, tf, nSamples)

    ODEtime = timer()
    t , numericResult = ode87(system,[t0,tf],IC,t,rtol=1e-9,atol=1e-12)
    ODEtime.toc()

    output_seq = numericResult

    # hyperparameters
    n_epochs = 5
    lr = 0.001
    input_size = problemDim
    output_size = problemDim
    num_layers = 1
    lookback = 1
    p_motion_knowledge = 1/numPeriods

    train_size = int(len(output_seq) * p_motion_knowledge)
    test_size = len(output_seq) - train_size

    train_in,train_out,test_in,test_out = create_datasets(output_seq,1,train_size,device)

    loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

    # initilizing the model, criterion, and optimizer for the data
    config = MambaConfig(d_model=problemDim, n_layers=num_layers)

    def returnModel(modelString = 'Mamba'):
        if modelString == 'Mamba':
            model = Mamba(config).to(device).double()
        elif modelString == 'LSTM':
            model = LSTM(input_size,30,output_size,num_layers,0).double().to(device)
        elif modelString == "lstmSA":
            model = LSTMSelfAttentionNetwork(input_size,30,output_size,num_layers,0).double().to(device)
        return model

    model = returnModel(modelString)

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = F.smooth_l1_loss
    criterion = torch.nn.HuberLoss()

    trainModel(model,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutAcc = True,printOutToc = True)
    t = t / tEnd

    networkPrediction = plotStatePredictions(model,t,output_seq,train_in,test_in,train_size,test_size,DU=DU,TU=TU,timeLabel='Periods')
    output_seq = nonDim2Dim6(output_seq,DU,TU)

    plotCR3BPPhasePredictions(output_seq,networkPrediction,L=None,earth=False,moon=False)
    plotCR3BPPhasePredictions(output_seq,networkPrediction,L=None,plane='xz',earth=False,moon=False)
    plotCR3BPPhasePredictions(output_seq,networkPrediction,L=None,plane='yz',earth=False,moon=False)
    plot3dCR3BPPredictions(output_seq,networkPrediction,L=None,earth=False,moon=False,networkLabel=modelString)


    newPlotSolutionErrors(output_seq,networkPrediction,t,timeLabel='Periods',percentError=True,states = ['x', 'y', 'z', '$\dot{x}$', '$\dot{y}$', '$\dot{z}$'])


    plotEnergy(output_seq,networkPrediction,t,jacobiConstant6,xLabel='Number of Periods (T)',yLabel='Jacobi Constant',nonDim=dim2NonDim6,DU = DU, TU = TU,networkLabel="Mamba")
    # # TRANSFER LEARN

    x_0,tEnd = CR3BPIC_target()

    IC = np.array(x_0)

    numPeriods = 5

    t0 = 0; tf = numPeriods * tEnd

    delT = 0.001
    nSamples = int(np.ceil((tf - t0) / delT))
    t = np.linspace(t0, tf, nSamples)

    t , numericResult = ode87(system,[t0,tf],IC,t,rtol=1e-15,atol=1e-15)

    output_seq = numericResult

    n_epochs = 2
    lr = 0.001
    input_size = problemDim
    output_size = problemDim
    num_layers = 1
    lookback = 1

    p_motion_knowledge = 1/numPeriods

    train_size = int(len(output_seq) * p_motion_knowledge)
    test_size = len(output_seq) - train_size

    train_in,train_out,test_in,test_out = create_datasets(output_seq,1,train_size,device)

    loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

    # initilizing the model, criterion, and optimizer for the data
    config = MambaConfig(d_model=problemDim, n_layers=num_layers)
    newModel = Mamba(config).to(device).double()

    newModel = returnModel(modelString)

    if modelString == "mamba":
        newModel = transferMamba(model,newModel,[True,True,False])
    elif modelString == "LSTM":
        newModel = transferLSTM(model,newModel)
    elif modelString == "lstmSA":
        newModel = transferModelAll(model,newModel)

    # newModel = LSTMSelfAttentionNetwork(input_size,50,output_size,num_layers,0).double().to(device)

    optimizer = torch.optim.Adam(newModel.parameters(),lr=lr)
    criterion = F.smooth_l1_loss

    trainModel(model,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutAcc = True,printOutToc = True)
    t = t / tEnd

    networkPrediction_target = plotStatePredictions(model,t,output_seq,train_in,test_in,train_size,test_size,DU=DU,TU=TU,timeLabel='Periods')

    output_seq = nonDim2Dim6(output_seq,DU,TU)

    plotCR3BPPhasePredictions(output_seq,networkPrediction_target,L=None,earth=False,moon=False)
    plotCR3BPPhasePredictions(output_seq,networkPrediction_target,L=None,plane='xz',earth=False,moon=False)
    plotCR3BPPhasePredictions(output_seq,networkPrediction_target,L=None,plane='yz',earth=False,moon=False)
    plot3dCR3BPPredictions(output_seq,networkPrediction_target,L=None,earth=False,moon=False,networkLabel=modelString)

    newPlotSolutionErrors(output_seq,networkPrediction_target,t,timeLabel='Periods',percentError=True,states = ['x', 'y', 'z', '$\dot{x}$', '$\dot{y}$', '$\dot{z}$'])

    plotEnergy(output_seq,networkPrediction_target,t,jacobiConstant6,xLabel='Number of Periods (T)',yLabel='Jacobi Constant',nonDim=dim2NonDim6,DU = DU, TU = TU,networkLabel="LSTM")


    errorAvg = np.nanmean(abs(networkPrediction_target-output_seq), axis=0)
    print("Average values of each dimension:")
    for i, avg in enumerate(errorAvg, 1):
        print(f"Dimension {i}: {avg}")

    torchinfo.summary(model)
    torchinfo.summary(newModel)

    printoutMaxLayerWeight(model)
    printoutMaxLayerWeight(newModel)

    plt.figure()
    plotSuperWeight(model,newPlot=False)
    plotSuperWeight(newModel,newPlot=False)
    plt.grid()
    plt.tight_layout()

    plt.figure()
    plotMinWeight(model,newPlot=False)
    plotMinWeight(newModel,newPlot=False)
    plt.grid()
    plt.tight_layout()

    del model
    del newModel

    modelString = "LSTM"
if plotOn is True:
    plt.show()

