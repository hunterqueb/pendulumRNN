import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo
from scipy.io import loadmat,savemat

from qutils.integrators import ode45
from qutils.plot import plotCR3BPPhasePredictions,plotOrbitPredictions, plotSolutionErrors
from qutils.ml import getDevice,printModelParmSize
from qutils.mlExtras import findDecAcc,generateTrajectoryPrediction
from qutils.orbital import nonDim2Dim4
from qutils.mamba import Mamba,MambaConfig

from nets import create_dataset

device = getDevice()

pathToData = 'matlab/1dPDF/'
fileName = 'asym_steeper1D_pdf_cdf_400'
fileExtention = '.mat'

pdf_approx_coeff = loadmat(pathToData+'asym_steeper1D_pdf_coeffs_400'+fileExtention)
cdf_approx_coeff = loadmat(pathToData+'asym_steeper1D_pdf_cumInteg_400'+fileExtention)



pdfSet = ['norm_cheb_rbf_coeff','norm_equi_rbf_coeff','norm_halton_rbf_coeff']
cdfSet = ['cheb_cumInteg','equi_cumInteg','halton_cumInteg']
combinedSet = ["cheb","equi","halton"]

learningSet = {}
for i in range(3):
    concatenated_array = np.concatenate((pdf_approx_coeff[pdfSet[i]], cdf_approx_coeff[cdfSet[i]]))
    learningSet[combinedSet[i]] = concatenated_array

# learningSet = ['cheb_rbf_coeff']
# learningSet = ['equi_rbf_coeff']
# learningSet = ['halton_rbf_coeff']

# learningSet = ['norm_cheb_rbf_coeff']
# learningSet = ['norm_equi_rbf_coeff']
# learningSet = ['norm_halton_rbf_coeff']

all_data = {}

for set in learningSet:

    learningSeq =  learningSet[set].T

    sequenceLength = learningSeq.shape[0]
    problemDim = learningSeq.shape[1]

    # hyperparameters
    n_epochs = 10
    # lr = 0.0007
    lr = 0.01
    input_size = problemDim
    output_size = problemDim
    num_layers = 1
    lookback = 1


    p_motion_knowledge = 1/4
    # train_size = 2
    train_size = int(sequenceLength * p_motion_knowledge)
    test_size = sequenceLength - train_size


    train = learningSeq[:train_size]
    test = learningSeq[train_size:]

    train_in,train_out = create_dataset(train,device,lookback=lookback)
    test_in,test_out = create_dataset(test,device,lookback=lookback)

    # testing can be the final matrix

    loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

    # initilizing the model, criterion, and optimizer for the data
    # config = MambaConfig(d_model=problemDim, n_layers=num_layers,d_conv=256)
    config = MambaConfig(d_model=problemDim, n_layers=num_layers,d_conv=32,expand_factor=4)
    model = Mamba(config).to(device).double()


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
            decAcc, err2 = findDecAcc(test_out,y_pred_test,printOut=False)
            err = np.concatenate((err1,err2),axis=0)

    model.eval()

    # data_in = torch.tensor(learningSeq[0,:].reshape(1,1,problemDim),device=device)
    # data_out_array = np.empty((1, int(tf/dt), problemDim))
    # data_out_array[0,0,:] = learningSeq[0,:].reshape(1,1,problemDim)
    
    with torch.no_grad():
    #     for i in range(int(tf/dt)-1):
    #         data_out = model(data_in)
    #         data_out_array[0,i+1,:] = data_out.cpu().numpy()
    #         data_in = data_out


    # finalData = np.squeeze(data_out_array)

        train_pred = np.ones_like(learningSeq) * np.nan
        train_pred[lookback:train_size] = model(train_in)[:,-1,:].cpu()
        train_pred[0,:] = learningSeq[0,:]

        test_pred = np.ones_like(learningSeq) * np.nan
        test_pred[train_size:sequenceLength-1] = model(test_in)[:, -1, :].cpu()
        data_in = torch.tensor(test_pred[-2,:].reshape(1,1,problemDim),device=device)
        test_pred[-1,:] = model(data_in)[:,-1,:].cpu().numpy()
    finalData = generateTrajectoryPrediction(train_pred,test_pred)

    all_data['pred_' + set] = finalData.T

    # savemat('matlab/1dPDF/prediction/' + set + '_pred.mat',{set + '_pred': finalData})

    predictedCoeffNorm = finalData
    trueCoffNorm = learningSet[set].T


    error = predictedCoeffNorm - trueCoffNorm
    errorAvg = np.nanmean(abs(error))
    
    print('Average error in for ' + set + ':',errorAvg)
    print("\ttrain loss %.4f, test loss %.4f\n" % (train_loss, test_loss))
torchinfo.summary(model,input_size=(1,1,problemDim))
printModelParmSize(model)
savemat(pathToData + fileName + '_pred' + fileExtention, all_data)

# # save the final y_pred_test


# # savemat('matlab/lowerDataMamba/prediction/normalized_pdf_tf.mat',{'normalized_pdf_tf': predictedPDFValues})
# 