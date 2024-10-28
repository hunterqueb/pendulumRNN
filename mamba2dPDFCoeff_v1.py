import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo
from scipy.io import loadmat,savemat
import matplotlib.lines as mlines

from qutils.integrators import ode45
from qutils.plot import plotCR3BPPhasePredictions,plotOrbitPredictions, plotSolutionErrors
from qutils.ml import getDevice,printModelParmSize
from qutils.mlExtras import findDecAcc,generateTrajectoryPrediction
from qutils.orbital import nonDim2Dim4
from qutils.mamba import Mamba,MambaConfig
from qutils.tictoc import timer

from nets import create_dataset

device = getDevice()

fileLocation = 'matlab/DDDAS-2d/newData/'
fileName = ['duff2D_2xUncer_pdfCoeff_del_t_0_1_tf_2_4_mamba',"duff2D_2xUncer_pdfCoeff_del_t_0_01_tf_2_4_mamba","duff2D_2xUncer_pdfCoeff_del_t_0_1_tf_4_8_mamba","duff2D_2xUncer_pdfCoeff_del_t_0_01_tf_4_8_mamba","linOsc_pdfCoeff_del_t_0_1_tf_2_4_mamba",'linOsc_pdfCoeff_del_t_0_01_tf_2_4_mamba',"linOsc_pdfCoeff_del_t_0_1_tf_4_8_mamba","linOsc_pdfCoeff_del_t_0_01_tf_4_8_mamba"]
fileName = ["duff2D_2xUncer_pdfCoeff_del_t_0_1_tf_4_8_mamba","duff2D_2xUncer_pdfCoeff_del_t_0_01_tf_4_8_mamba","linOsc_pdfCoeff_del_t_0_1_tf_4_8_mamba","linOsc_pdfCoeff_del_t_0_01_tf_4_8_mamba"]
fileExtension = '.mat'

for file in fileName:

    pdf_approx_coeff = loadmat(fileLocation+file+fileExtension)

    learningSet = ['double_norm_coeff_half','double_norm_coeff_quarter']

    # learningSet = ['cheb_rbf_coeff']
    # learningSet = ['equi_rbf_coeff']
    # learningSet = ['halton_rbf_coeff']

    # learningSet = ['norm_cheb_rbf_coeff']
    # learningSet = ['norm_equi_rbf_coeff']
    # learningSet = ['norm_halton_rbf_coeff']

    all_data = {}

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

        if set == learningSet[0]:
            p_motion_knowledge = 1/2
        else:
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

            # print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))

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
        trueCoffNorm = pdf_approx_coeff[set].T


        error = predictedCoeffNorm - trueCoffNorm
        errorAvg = np.nanmean(abs(error))
        
        print('Average error in for ' + set + ':',errorAvg)
        print("\ttrain loss %.4f, test loss %.4f\n" % (train_loss, test_loss))

        # train_plot = np.empty(learningSeq.shape)
        # train_plot[:] = np.nan
        train_plot = np.pad(train, ((0, learningSeq.shape[0] - train.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
        test_plot = np.pad(predictedCoeffNorm, ((0, learningSeq.shape[0] - predictedCoeffNorm.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)

        
        t = pdf_approx_coeff['tspan'].T
        plt.figure()
        for i in range(problemDim):
            plt.plot(t,learningSeq[:,i])
        plt.plot(t,train_plot,'k', linestyle='dashed',label='_nolegend_')
        plt.plot(t,test_plot,'grey', linestyle=':',label='_nolegend_')
        training_region_line = mlines.Line2D([], [], color='k', linestyle='dashed', label='Training Region')
        test_region_line = mlines.Line2D([], [], color='grey', linestyle=':', label='Prediction')
        truth_region_line = mlines.Line2D([], [], color='blue', label='Truth')
        plt.legend(handles=[training_region_line,test_region_line,truth_region_line])
        plt.grid()
        plt.xlabel('Time (sec)')
        plt.ylabel('RBF Coefficent Values (none)')
        plt.show()

    torchinfo.summary(model,input_size=(1,1,problemDim))
    printModelParmSize(model)
    savemat(fileLocation+file+'_pred'+fileExtension, all_data)

# # save the final y_pred_test


# # savemat('matlab/lowerDataMamba/prediction/normalized_pdf_tf.mat',{'normalized_pdf_tf': predictedPDFValues})
# 