import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo

from scipy.io import loadmat,savemat

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from qutils.ml import getDevice,printModelParmSize, create_datasets, transferMamba
from qutils.mlExtras import findDecAcc,generateTrajectoryPrediction
from qutils.mamba import Mamba,MambaConfig
from qutils.tictoc import timer

device = getDevice()

fileLocation = 'scripts/mamba/nsin/'
file = "lin_forced_osc_pdfCoeff_del_t_0_01_tf_4_8_mamba"
file = "lin_sin_forced_osc_pdfCoeff_del_t_0_01_tf_4_8_mamba"
# file = "lin_forced_osc_pdfCoeff_del_t_0_01_tf_25_mamba"
# file = "lin_sin_forced_osc_pdfCoeff_del_t_0_01_tf_25_mamba"
fileExtension = ".mat"
pdf_approx_coeff = loadmat(fileLocation+file+fileExtension)

learningSet = ['double_norm_coeff_half','double_norm_coeff_quarter']
all_data = {}

def makePred(model,learningSeq):
    model.eval()

    testTime = timer()

    with torch.no_grad():
        train_pred = np.ones_like(learningSeq) * np.nan
        train_pred[seq_length:train_size+seq_length] = model(train_in)[:,-1,:].cpu()
        train_pred[0,:] = learningSeq[0,:]

        test_pred = np.ones_like(learningSeq) * np.nan
        test_pred[train_size+seq_length:] = model(test_in)[:, -1, :].cpu()
        data_in = torch.tensor(test_pred[-2,:].reshape(1,1,problemDim),device=device)
        test_pred[-1,:] = model(data_in)[:,-1,:].cpu().numpy()
    finalData = generateTrajectoryPrediction(train_pred,test_pred)

    testTime.toc()

    return finalData

for set in learningSet:
    learningSeq =  pdf_approx_coeff[set].T

    sequenceLength = learningSeq.shape[0]
    problemDim = learningSeq.shape[1]

    # hyperparameters
    n_epochs = 20
    # lr = 0.0007
    lr = 0.01
    input_size = problemDim
    output_size = problemDim
    num_layers = 1
    lookback = 1
    seq_length = 1

    if set == learningSet[0]:
        p_motion_knowledge = 1/2
    else:
        p_motion_knowledge = 1/4
    # train_size = 2
    train_size = int(sequenceLength * p_motion_knowledge)
    test_size = sequenceLength - train_size


    train = learningSeq[:train_size]
    test = learningSeq[train_size:]

    train_in,train_out,test_in,test_out = create_datasets(learningSeq,seq_length,train_size,device)

    # testing can be the final matrix

    loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

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

    unforced_pred = makePred(model,learningSeq)


    newModel = Mamba(config).to(device).double()
    newModel = transferMamba(model,newModel)
    

    learningSeq =  pdf_approx_coeff[set+"_forced"].T

    sequenceLength = learningSeq.shape[0]
    problemDim = learningSeq.shape[1]
    n_epochs = 2


    train_size = int(sequenceLength * p_motion_knowledge)
    test_size = sequenceLength - train_size

    train = learningSeq[:train_size]
    test = learningSeq[train_size:]

    train_in,train_out,test_in,test_out = create_datasets(learningSeq,seq_length,train_size,device)

    # testing can be the final matrix

    loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

    optimizer = torch.optim.Adam(newModel.parameters(),lr=lr)
    criterion = F.smooth_l1_loss

    trainTime = timer()

    for epoch in range(n_epochs):

        # trajPredition = plotPredition(epoch,model,'target',t=t*TU,output_seq=pertNR)

        newModel.train()
        for X_batch, y_batch in loader:
            y_pred = newModel(X_batch)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        newModel.eval()
        with torch.no_grad():
            y_pred_train = newModel(train_in)
            train_loss = np.sqrt(criterion(y_pred_train, train_out).cpu())
            y_pred_test = newModel(test_in)
            test_loss = np.sqrt(criterion(y_pred_test, test_out).cpu())

            decAcc, err1 = findDecAcc(train_out,y_pred_train,printOut=False)
            decAcc, err2 = findDecAcc(test_out,y_pred_test,printOut=False)
            err = np.concatenate((err1,err2),axis=0)

    print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))

    trainTime.toc()

    forced_pred = makePred(newModel,learningSeq)

    t = pdf_approx_coeff['tspan'].T

    unforcedTrueCoffNorm = pdf_approx_coeff[set].T
    forcedTrueCoffNorm = pdf_approx_coeff[set+"_forced"].T




    # TODO - plot the coefficent trajectories


    def plotCoeffPred(finalData,trueCoffNorm):
        predictedCoeffNorm = finalData
        error = predictedCoeffNorm - trueCoffNorm
        errorAvg = np.nanmean(abs(error))
    
        print('Average error in for ' + set + ':',errorAvg)
        print("\ttrain loss %.4f, test loss %.4f\n" % (train_loss, test_loss))

        train_plot = np.pad(train, ((0, learningSeq.shape[0] - train.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
        test_plot = np.pad(predictedCoeffNorm, ((0, learningSeq.shape[0] - predictedCoeffNorm.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)

        for i in range(problemDim):
            plt.plot(t,learningSeq[:,i])
        plt.plot(t,learningSeq[:,0])

        plt.plot(t,train_plot,'k', linestyle='dashed',label='_nolegend_')
        plt.plot(t,test_plot,'grey', linestyle=':',label='_nolegend_')
        plt.grid()
        plt.xlabel('Time (sec)')
        plt.ylabel('RBF Coefficent Values (none)')

    def plotCoeffPredAtIndex(i,finalData,trueCoffNorm):
        predictedCoeffNorm = finalData
        error = predictedCoeffNorm - trueCoffNorm
        errorAvg = np.nanmean(abs(error))
    
        print('Average error in for ' + set + ':',errorAvg)
        print("\ttrain loss %.4f, test loss %.4f\n" % (train_loss, test_loss))

        train_plot = np.pad(train, ((0, learningSeq.shape[0] - train.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
        test_plot = np.pad(predictedCoeffNorm, ((0, learningSeq.shape[0] - predictedCoeffNorm.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)

        plt.plot(t,learningSeq[:,i])

        plt.plot(t,train_plot[:,i],'k', linestyle='dashed',label='_nolegend_')
        plt.plot(t,test_plot[:,i],'grey', linestyle=':',label='_nolegend_')
        plt.grid()
        plt.xlabel('Time (sec)')
        plt.ylabel('RBF Coefficent Values (none)')


    # plt.figure()
    # plotCoeffPred(unforced_pred,unforcedTrueCoffNorm)
    # plotCoeffPred(forced_pred,forcedTrueCoffNorm)
    # training_region_line = mlines.Line2D([], [], color='k', linestyle='dashed', label='Training Region')
    # test_region_line = mlines.Line2D([], [], color='grey', linestyle=':', label='Prediction')
    # truth_region_line = mlines.Line2D([], [], color='blue', label='Truth')
    # plt.legend(handles=[training_region_line,test_region_line,truth_region_line])
    # plt.show()

    plt.figure()
    plotCoeffPredAtIndex(1,unforced_pred,unforcedTrueCoffNorm)
    # plotCoeffPredAtIndex(1,forced_pred,forcedTrueCoffNorm)
    training_region_line = mlines.Line2D([], [], color='k', linestyle='dashed', label='Training Region')
    test_region_line = mlines.Line2D([], [], color='grey', linestyle=':', label='Prediction')
    truth_region_line = mlines.Line2D([], [], color='blue', label='Truth')
    plt.legend(handles=[training_region_line,test_region_line,truth_region_line])
    plt.show()