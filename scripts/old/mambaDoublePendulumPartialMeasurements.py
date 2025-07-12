import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data

from qutils.integrators import ode45
from qutils.plot import plotCR3BPPhasePredictions,plotOrbitPredictions, plotSolutionErrors
from qutils.ml.utils import findDecAcc,getDevice
from qutils.orbital import nonDim2Dim4

from qutils.ml.regression import create_datasets as create_dataset, LSTMSelfAttentionNetwork,genPlotPrediction
from qutils.mambaAtt import Mamba, MambaConfig


device = getDevice()

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

t , numericResult = ode45(doublePendulumODE,[tStart,tEnd],initialConditions,tSpanExplicit)

output_seq = numericResult

# hyperparameters
n_epochs = 50
# lr = 5*(10**-5)
# lr = 0.85
lr = 0.8
lr = 0.08
lr = 0.004
lr = 0.0001
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = 1
# p_motion_knowledge = 0.5


# train_size = int(len(output_seq) * p_motion_knowledge)
train_size = 2
test_size = len(output_seq) - train_size

train1, test = output_seq[:train_size], output_seq[train_size:]

randomNum = int(np.random.rand() * len(output_seq))
train2 = output_seq[randomNum:randomNum+2]

train_in1,train_out1 = create_dataset(train1,device,lookback=lookback)
train_in2,train_out2 = create_dataset(train2,device,lookback=lookback)

train_in = torch.cat((train_in1, train_in2), dim=0)
train_out = torch.cat((train_out1, train_out2), dim=0)

test_in,test_out = create_dataset(test,device,lookback=lookback)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

# initilizing the model, criterion, and optimizer for the data
config = MambaConfig(d_model=problemDim, n_layers=num_layers)
model = Mamba(config).to(device).double()
# model = LSTMSelfAttentionNetwork(input_size,50,output_size,num_layers,0).double().to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = F.smooth_l1_loss

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
        train_plot, test_plot = genPlotPrediction(model,output_seq,train_in,test_in,train_size,1)

        # output_seq = nonDim2Dim4(output_seq)
        # train_plot = nonDim2Dim4(train_plot)
        # test_plot = nonDim2Dim4(test_plot)
    
        fig, axes = plt.subplots(2,2)

        axes[0,0].plot(t,output_seq[:,0], c='b',label = 'True Motion')
        axes[0,0].plot(t,train_plot[:,0], c='r',label = 'Training Region')
        axes[0,0].plot(t,test_plot[:,0], c='g',label = 'Predition')
        axes[0,0].set_xlabel('time (sec)')
        axes[0,0].set_ylabel('theta1 (rad)')

        axes[0,1].plot(t,output_seq[:,1], c='b',label = 'True Motion')
        axes[0,1].plot(t,train_plot[:,1], c='r',label = 'Training Region')
        axes[0,1].plot(t,test_plot[:,1], c='g',label = 'Predition')
        axes[0,1].set_xlabel('time (sec)')
        axes[0,1].set_ylabel('theta1dot (rad/s)')

        axes[1,0].plot(t,output_seq[:,2], c='b',label = 'True Motion')
        axes[1,0].plot(t,train_plot[:,2], c='r',label = 'Training Region')
        axes[1,0].plot(t,test_plot[:,2], c='g',label = 'Predition')
        axes[1,0].set_xlabel('time (sec)')
        axes[1,0].set_ylabel('theta2 (rad)')

        axes[1,1].plot(t,output_seq[:,3], c='b',label = 'True Motion')
        axes[1,1].plot(t,train_plot[:,3], c='r',label = 'Training Region')
        axes[1,1].plot(t,test_plot[:,3], c='g',label = 'Predition')
        axes[1,1].set_xlabel('time (sec)')
        axes[1,1].set_ylabel('theta2dot (rad/s)')


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

plotSolutionErrors(output_seq,networkPrediction,t)
# plotDecAccs(decAcc,t,problemDim)
errorAvg = np.nanmean(abs(networkPrediction-output_seq) * 90 / np.pi, axis=0)
print("Average error of each dimension:")
unitLabels = ['deg','deg/s','deg','deg/s']
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg} {unitLabels[i-1]}")


if plotOn is True:
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(output_seq[:,0],output_seq[:,2],'r',label = "Truth")
    plt.plot(networkPrediction[:,0],networkPrediction[:,2],'b',label = "NN")
    plt.xlabel('Theta 1')
    plt.ylabel('Theta 1 Dot')
    plt.axis('equal')


    plt.subplot(2, 1, 2)
    plt.plot(output_seq[:,1],output_seq[:,3],'r',label = "Truth")
    plt.plot(networkPrediction[:,1],networkPrediction[:,3],'b',label = "NN")
    plt.xlabel('Theta 2')
    plt.ylabel('Theta 2 Dot')
    plt.axis('equal')
    plt.legend()

    plt.show()


