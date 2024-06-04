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
from qutils.tictoc import timer

from nets import create_dataset

device = getDevice()

fileLocation = 'matlab/DDDAS-2d/'
fileName1 = 'duff2D_conserv_pdfCoeff_mamba'
fileName2 = 'duff2D_doubleUncer_pdfCoeff_mamba'
fileExtension = '.mat'

pdf_approx_coeff_1 = loadmat(fileLocation+fileName1+fileExtension)
pdf_approx_coeff_2 = loadmat(fileLocation+fileName2+fileExtension)

learningSet = ['double_norm_coeff_half','double_norm_coeff_quarter']

# learningSet = ['cheb_rbf_coeff']
# learningSet = ['equi_rbf_coeff']
# learningSet = ['halton_rbf_coeff']

# learningSet = ['norm_cheb_rbf_coeff']
# learningSet = ['norm_equi_rbf_coeff']
# learningSet = ['norm_halton_rbf_coeff']

all_data = {}

for set in learningSet:
    learningSeq1 =  pdf_approx_coeff_1[set].T
    learningSeq2 =  pdf_approx_coeff_2[set].T

    learningSeq = np.dstack((learningSeq1,learningSeq2))

    sequenceLength = learningSeq.shape[0]
    problemDim = learningSeq.shape[1]
    setSize = learningSeq.shape[2]

    # hyperparameters
    n_epochs = 20
    # lr = 0.0007
    lr = 0.01
    input_size = problemDim
    output_size = problemDim
    num_layers = 1
    lookback = 1

    if set == learningSet[0]:
        p_motion_knowledge = 1/2
    else:
        p_motion_knowledge = 1/4
    # train_size = 2
    train_size = int(sequenceLength * p_motion_knowledge)
    test_size = sequenceLength - train_size


    train1 = learningSeq1[:train_size]
    test1 = learningSeq1[train_size:]

    train_in1,train_out1 = create_dataset(train1,device,lookback=lookback)
    test_in1,test_out1 = create_dataset(test1,device,lookback=lookback)


    train2 = learningSeq2[:train_size]
    test2 = learningSeq2[train_size:]

    train_in2,train_out2 = create_dataset(train2,device,lookback=lookback)
    test_in2,test_out2 = create_dataset(test2,device,lookback=lookback)

    train_in = torch.concatenate((train_in1,train_in2),axis=1)
    train_out = torch.concatenate((train_out1,train_out2),axis=1)
    test_in = torch.concatenate((test_in1,test_in2),axis=1)
    test_out = torch.concatenate((test_out1,test_out2),axis=1)

    # testing can be the final matrix

    loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=100)

    # initilizing the model, criterion, and optimizer for the data
    # config = MambaConfig(d_model=problemDim, n_layers=num_layers,d_conv=256)
    config = MambaConfig(d_model=problemDim, n_layers=num_layers,d_conv=32,expand_factor=1)
    model = Mamba(config).to(device).double()


    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = F.smooth_l1_loss

    trainTime = timer()

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

        print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))

    trainTime.toc()
    
    model.eval()

    # testTime = timer()

    with torch.no_grad():
        train_pred = np.ones_like(learningSeq) * np.nan
        train_pred[lookback:train_size] = model(train_in)[:,-1,:].cpu()
        train_pred[0,:] = learningSeq[0,:]

        test_pred = np.ones_like(learningSeq) * np.nan
        test_pred[train_size:sequenceLength-1] = model(test_in)[:, -1, :].cpu()
        data_in = torch.tensor(test_pred[-2,:].reshape(1,1,problemDim),device=device)
        test_pred[-1,:] = model(data_in)[:,-1,:].cpu().numpy()
    finalData = generateTrajectoryPrediction(train_pred,test_pred)

    # testTime.toc()

    all_data[set + '_pred'] = finalData.T

    # savemat('matlab/1dPDF/prediction/' + set + '_pred.mat',{set + '_pred': finalData})

    predictedCoeffNorm = finalData
    # trueCoffNorm = pdf_approx_coeff[set].T
    # error = predictedCoeffNorm - trueCoffNorm
    # errorAvg = np.nanmean(abs(error))
    # print('Average error in for ' + set + ':',errorAvg)

    print("\ttrain loss %.4f, test loss %.4f\n" % (train_loss, test_loss))
torchinfo.summary(model,input_size=(1,1,problemDim))
printModelParmSize(model)
savemat(fileLocation+fileName1+'_pred'+fileExtension, all_data)

# # save the final y_pred_test


# # savemat('matlab/lowerDataMamba/prediction/normalized_pdf_tf.mat',{'normalized_pdf_tf': predictedPDFValues})
# 