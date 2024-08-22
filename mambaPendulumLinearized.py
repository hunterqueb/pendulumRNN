import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data

from qutils.integrators import ode45
from qutils.plot import plotCR3BPPhasePredictions,plotOrbitPredictions, plotSolutionErrors
from qutils.ml import printModelParmSize, printModelParameters, getDevice, Adam_mini
from qutils.mlExtras import findDecAcc
from qutils.orbital import nonDim2Dim4

from nets import create_dataset, LSTMSelfAttentionNetwork, LSTM, TransformerModel
from qutils.mamba import Mamba, MambaConfig

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

plotOn = True

problemDim = 2 

device = getDevice()

m = 1
l = 1
g = 9.81
parameters = np.array([m,l,g])

def pendulumODE(t,theta,p=parameters):
    # m = p[0]
    L = p[1]
    g = p[2]
    dtheta1 = theta[1]
    dtheta2 = -g/L*np.sin(theta[0])
    return np.array([dtheta1, dtheta2])

def pendulumLinearODE(t,theta,p=parameters):
    # linearized SSM
    # [0    1;
    #  -g/L 0]
    
    L = p[1]
    g = p[2]
    dtheta1 = theta[1]
    dtheta2 = -g/L*theta[0]
    return np.array([dtheta1, dtheta2])


theta1_0 = np.radians(80)
thetadot1_0 = np.radians(1)
initialConditions = np.array([theta1_0,thetadot1_0],dtype=np.float64)

# initialConditions = np.radians(np.random.uniform(-180, 180, (problemDim,)))

tStart = 0
tEnd = 20
tSpan = np.array([tStart,tEnd])
dt = 0.01
tSpanExplicit = np.linspace(tStart,tEnd,int(tEnd / dt))

t , numericResult = ode45(pendulumODE,[tStart,tEnd],initialConditions,tSpanExplicit)

output_seq = numericResult

# hyperparameters
n_epochs = 50
# lr = 5*(10**-5)
# lr = 0.85
lr = 0.8
lr = 0.08
lr = 0.004
# lr = 0.0001
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = 1
p_motion_knowledge = 0.5


train_size = int(len(output_seq) * p_motion_knowledge)
train_size = 20
test_size = len(output_seq) - train_size

train, test = output_seq[:train_size], output_seq[train_size:]

train_in,train_out = create_dataset(train,device,lookback=lookback)
test_in,test_out = create_dataset(test,device,lookback=lookback)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

# initilizing the model, criterion, and optimizer for the data
config = MambaConfig(d_model=problemDim, n_layers=num_layers,expand_factor=1,d_state=problemDim)
model = Mamba(config).to(device).double()
# model = LSTM(input_size,10,output_size,num_layers,0).double().to(device)

optimizer = Adam_mini(model,lr=lr)
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
        print(model.layers[0].mixer.A_SSM)
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

plotSolutionErrors(output_seq,networkPrediction,t,units='rad',states=('\\theta_1','\\theta_2'))
# plotDecAccs(decAcc,t,problemDim)
errorAvg = np.nanmean(abs(networkPrediction-output_seq) * 90 / np.pi, axis=0)
print("Average error of each dimension:")
unitLabels = ['deg','deg/s','deg','deg/s']
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg} {unitLabels[i-1]}")

# printModelParmSize(model)
# printModelParameters(model)

# print(model.layers[0].mixer.A_SSM.shape)
# print(model.layers[0].mixer.A_SSM)
# # print(pendulumLinearODE(0,initialConditions))
# print(model.layers[0].mixer.B_SSM.shape)
# print(model.layers[0].mixer.C_SSM.shape)

if plotOn is True:
    plt.figure()
    plt.plot(output_seq[:,0],output_seq[:,1],'r',label = "Truth")
    plt.plot(networkPrediction[:,0],networkPrediction[:,1],'b',label = "NN")
    plt.xlabel('Theta 1')
    plt.ylabel('Theta 1 Dot')
    plt.axis('equal')
    plt.grid()

    plt.show()


