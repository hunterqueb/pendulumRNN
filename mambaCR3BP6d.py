import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo

from qutils.integrators import ode1412, ode85
from qutils.plot import plotCR3BPPhasePredictions,plotOrbitPredictions, plotSolutionErrors
from qutils.mlExtras import findDecAcc
from qutils.orbital import nonDim2Dim6
from qutils.mamba import Mamba, MambaConfig
from qutils.ml import printModelParmSize, getDevice

from memory_profiler import profile

from nets import create_dataset, LSTMSelfAttentionNetwork

DEBUG = True
plotOn = True

problemDim = 6
m_1 = 5.974E24  # kg
m_2 = 7.348E22 # kg
mu = m_2/(m_1 + m_2)

# short period L4 "kidney bean"
# x_0 = 0.487849413
# y_0 = 1.471265959
# vx_0 = 1.024841387
# vy_0 = -0.788224219
# tEnd = 6.2858346244258847

# halo orbit around L1 - id 754
x_0 = 8.4292851057904816E-1
y_0 = -2.8183883963983186E-24
z_0 = 5.0648489866815749E-1
vx_0 = -2.1104238512337928E-12
vy_0 = 1.5397481970828469E-1
vz_0 = 5.4929593265750457E-12
tEnd = 2.7783178612577766E+0

# halo around l3 - id 10
x_0 = -1.6775753144556563E-1
y_0 = -1.1367134367998278E-25
z_0 = 1.9153870456961195E+0
vx_0 = 7.3594870907425127E-11
vy_0 = 1.2311060080547762E-1
vz_0 = -6.4297898960019798E-11
tEnd = 5.9550558101971349E+0

# butterfly id 270
x_0 = 1.0396500770783366E+0
y_0 = -1.9627997004828110E-27
z_0 = 2.5485125128643876E-1
vx_0 = -4.8969667549929713E-14
vy_0 = -2.8617373429417120E-1
vz_0 = -1.0568735643727322E-13
tEnd = 9.2235616765684298E+0

# dragonfly id 71
x_0 =  1.1442729375808927E+0
y_0 =  -2.5570518644231412E-20
z_0 =  9.7411817634090236E-2
vx_0 = 1.0072600603243715E-2
vy_0 = -3.4395156291560752E-1
vz_0 = 2.8345900193831092E-1
tEnd = 6.9983567996146689E+0

# lyapunov id 312
x_0 = 5.9335219082124890E-1
y_0 = 1.9799386716596461E-23
z_0 = -4.4836817539948721E-26
vx_0 = -1.0419834306319422E-13
vy_0 = 8.9233311603791643E-1
vz_0 = 6.4404464538717388E-25
tEnd = 6.9604405499234705E+0

# vSquared = (vx_0**2 + vy_0**2)
# xn1 = -mu
# xn2 = 1-mu
# rho1 = np.sqrt((x_0-xn1)**2+y_0**2)
# rho2 = np.sqrt((x_0-xn2)**2+y_0**2)

# C0 = (x_0**2 + y_0**2) + 2*(1-mu)/rho1 + 2*mu/rho2 - vSquared
# print('Jacobi Constant: {}'.format(C0))

# Then stack everything together into the state vector
r_0 = np.array((x_0, y_0,z_0))
v_0 = np.array((vx_0, vy_0,vz_0))
x_0 = np.hstack((r_0, v_0))


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

IC = np.array(x_0)


device = getDevice()


numPeriods = 5


t0 = 0; tf = numPeriods * tEnd

delT = 0.001
nSamples = int(np.ceil((tf - t0) / delT))
t = np.linspace(t0, tf, nSamples)

# t , numericResult = ode1412(system,[t0,tf],IC,t)
t , numericResult = ode85(system,[t0,tf],IC,t)

t = t / tEnd

output_seq = numericResult

# hyperparameters
n_epochs = 1
# lr = 5*(10**-5)
# lr = 0.85
lr = 0.8
lr = 0.08
lr = 0.004
lr = 0.001
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = 1
# p_motion_knowledge = 0.5
p_motion_knowledge = 1/numPeriods


train_size = int(len(output_seq) * p_motion_knowledge)
test_size = len(output_seq) - train_size

train, test = output_seq[:train_size], output_seq[train_size:]

train_in,train_out = create_dataset(train,device,lookback=lookback)
test_in,test_out = create_dataset(test,device,lookback=lookback)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

# initilizing the model, criterion, and optimizer for the data
config = MambaConfig(d_model=problemDim, n_layers=num_layers,d_conv=16)
@profile
def returnModel():
    model = Mamba(config).to(device).double()
    return model

model = returnModel()
# model = LSTMSelfAttentionNetwork(input_size,50,output_size,num_layers,0).double().to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = F.smooth_l1_loss
criterion = torch.nn.HuberLoss()

for epoch in range(n_epochs):

    # trajPredition = plotPredition(epoch,model,'target',t=t*TU,output_seq=pertNR)

    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    with torch.no_grad():
        y_pred_train = model(train_in)
        train_loss = np.sqrt(criterion(y_pred_train, train_out).cpu())
        y_pred_test = model(test_in)
        test_loss = np.sqrt(criterion(y_pred_test, test_out).cpu())

        decAcc, err1 = findDecAcc(train_out,y_pred_train,printOut=False)
        decAcc, err2 = findDecAcc(test_out,y_pred_test)
        err = np.concatenate((err1,err2),axis=0)

    print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))




def plotPredition(epoch,model,trueMotion,prediction='source',err=None):
        output_seq = trueMotion
        with torch.no_grad():
            # shift train predictions for plotting
            train_plot = np.ones_like(output_seq) * np.nan
            y_pred = model(train_in)
            y_pred = y_pred[:, -1, :]
            train_plot[lookback:train_size] = model(train_in)[:, -1, :].cpu()
            # shift test predictions for plotting
            test_plot = np.ones_like(output_seq) * np.nan
            test_plot[train_size+lookback:len(output_seq)] = model(test_in)[:, -1, :].cpu()

        # output_seq = nonDim2Dim4(output_seq)
        # train_plot = nonDim2Dim4(train_plot)
        # test_plot = nonDim2Dim4(test_plot)
    
        fig, axes = plt.subplots(2,3)

        axes[0,0].plot(t,output_seq[:,0], c='b',label = 'True Motion')
        axes[0,0].plot(t,train_plot[:,0], c='r',label = 'Training Region')
        axes[0,0].plot(t,test_plot[:,0], c='g',label = 'Predition')
        axes[0,0].set_xlabel('time (sec)')
        axes[0,0].set_ylabel('x (km)')

        axes[0,1].plot(t,output_seq[:,1], c='b',label = 'True Motion')
        axes[0,1].plot(t,train_plot[:,1], c='r',label = 'Training Region')
        axes[0,1].plot(t,test_plot[:,1], c='g',label = 'Predition')
        axes[0,1].set_xlabel('time (sec)')
        axes[0,1].set_ylabel('y (km)')

        axes[0,2].plot(t,output_seq[:,2], c='b',label = 'True Motion')
        axes[0,2].plot(t,train_plot[:,2], c='r',label = 'Training Region')
        axes[0,2].plot(t,test_plot[:,2], c='g',label = 'Predition')
        axes[0,2].set_xlabel('time (sec)')
        axes[0,2].set_ylabel('z (km)')

        axes[1,0].plot(t,output_seq[:,0], c='b',label = 'True Motion')
        axes[1,0].plot(t,train_plot[:,0], c='r',label = 'Training Region')
        axes[1,0].plot(t,test_plot[:,0], c='g',label = 'Predition')
        axes[1,0].set_xlabel('time (sec)')
        axes[1,0].set_ylabel('xdot (km/s)')

        axes[1,1].plot(t,output_seq[:,1], c='b',label = 'True Motion')
        axes[1,1].plot(t,train_plot[:,1], c='r',label = 'Training Region')
        axes[1,1].plot(t,test_plot[:,1], c='g',label = 'Predition')
        axes[1,1].set_xlabel('time (sec)')
        axes[1,1].set_ylabel('ydot (km/s)')

        axes[1,2].plot(t,output_seq[:,2], c='b',label = 'True Motion')
        axes[1,2].plot(t,train_plot[:,2], c='r',label = 'Training Region')
        axes[1,2].plot(t,test_plot[:,2], c='g',label = 'Predition')
        axes[1,2].set_xlabel('time (sec)')
        axes[1,2].set_ylabel('zdot (km/s)')


        plt.legend(loc='upper left', bbox_to_anchor=(1,0.5))
        plt.tight_layout()

        if prediction == 'source':
            plt.savefig('predict/predict%d.png' % epoch)
        if prediction == 'target':
            plt.savefig('predict/newPredict%d.png' % epoch)
        plt.close()

        if err is not None:
            fig, (ax1, ax2) = plt.subplots(2,1)
            ax1.plot(err[:,0:2],label=('x','y'))
            ax1.set_xlabel('node #')
            ax1.set_ylabel('error (km)')
            ax1.legend()
            ax2.plot(err[:,2:4],label=('xdot','ydot'))
            ax2.set_xlabel('node #')
            ax2.set_ylabel('error (km/s)')
            ax2.legend()
            # ax2.plot(np.average(err,axis=0)*np.ones(err.shape))
            plt.show()
            
        trajPredition = np.zeros_like(train_plot)

        for i in range(test_plot.shape[0]):
            for j in range(test_plot.shape[1]):
                # Check if either of the matrices has a non-nan value at the current position
                if not np.isnan(test_plot[i, j]) or not np.isnan(train_plot[i, j]):
                    # Choose the non-nan value if one exists, otherwise default to test value
                    trajPredition[i, j] = test_plot[i, j] if not np.isnan(test_plot[i, j]) else train_plot[i, j]
                else:
                    # If both are nan, set traj element to nan
                    trajPredition[i, j] = np.nan

        return trajPredition

networkPrediction = plotPredition(epoch+1,model,output_seq)
plotCR3BPPhasePredictions(output_seq,networkPrediction,L=3)
plotCR3BPPhasePredictions(output_seq,networkPrediction,L=3,plane='xz')
plotCR3BPPhasePredictions(output_seq,networkPrediction,L=3,plane='yz')


DU = 384400
G = 6.67430e-11
TU = np.sqrt(DU**3 / (G*(m_1+m_2)))

networkPrediction = nonDim2Dim6(networkPrediction,DU,TU)
output_seq = nonDim2Dim6(output_seq,DU,TU)

# plotOrbitPredictions(output_seq,networkPrediction,t=t)
plotSolutionErrors(output_seq,networkPrediction,t,problemDim)
# plotDecAccs(decAcc,t,problemDim)
errorAvg = np.nanmean(abs(networkPrediction-output_seq), axis=0)
print("Average values of each dimension:")
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg}")


# torchinfo.summary(model)
printModelParmSize(model)
print('rk85 on 2 period halo orbit takes 1.199 MB of memory to solve')

if plotOn is True:
    plt.show()