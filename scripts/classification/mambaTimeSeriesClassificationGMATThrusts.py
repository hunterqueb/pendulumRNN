# parse at the beginning before long imports
# script usage

# call the script from the main folder directory, adding --save saves the output to a log file in the location of the datasets
# $ python scripts/classification/mambaTimeSeriesClassificationGMATThrusts.py \
# --systems 10000 --propMin 5 --OE --norm --orbit vleo 

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no-lstm',dest="use_lstm", action='store_false', help='Use LSTM model')
parser.add_argument("--systems", type=int, default=10000, help="Number of random systems to access")
parser.add_argument("--propMin", type=int, default=30, help="Minimum propagation time in minutes")
parser.add_argument("--orbit", type=str, default="vleo", help="Orbit type: vleo, leo")
parser.add_argument("--test", type=str, default=None, help="Orbit type for test set: vleo, leo, OR the same as --orbit and an integer number of random systems to use for testing")
parser.add_argument("--testSys", type=int, default=10000, help="Number of systems to use for testing if --test is a different string than --orbit")
parser.add_argument("--OE", action='store_true', help="Use OE elements instead of ECI states")
parser.add_argument("--noise", action='store_true', help="Add noise to the data")
parser.add_argument("--norm", action='store_true', help="Normalize the semi-major axis by Earth's radius")
parser.add_argument("--one-pass",dest="one_pass",action='store_true', help="Use one pass learning.")
parser.add_argument("--save",dest="save_to_log",action="store_true",help="output console printout to log file in the same location as datasets")
parser.add_argument("--energy",dest="use_energy",action="store_true",help="Use energy as a feature.")
parser.add_argument("--hybrid",dest="use_hybrid",action="store_true",help="Use a hybrid network.")
parser.add_argument("--superweight",dest="find_SW",action="store_true",help="Superweight analysis")
parser.add_argument("--no-classic",dest="use_classic",action="store_false",help="Use classic ML classification for comparison")
parser.add_argument("--nearest",dest="use_nearestNeighbor",action="store_true",help="Use classic ML classification (1-nearest neighbor w/ DTW) for comparison")
parser.add_argument('--saveNets', dest="saveNets",action='store_true', help='Save the trained networks. Saves to the same location as a saved log file.')
parser.add_argument('--classic', dest="old_classic",action='store_true', help='DO NOT USE. DUMMY ARGUMENT TO AVOID BREAKING OLD SCRIPTS.')

parser.set_defaults(use_lstm=True)
parser.set_defaults(OE=False)
parser.set_defaults(noise=False)
parser.set_defaults(norm=False)
parser.set_defaults(one_pass=False)
parser.set_defaults(save_to_log=False)
parser.set_defaults(use_energy=False)
parser.set_defaults(use_hybrid=False)
parser.set_defaults(find_SW=False)
parser.set_defaults(use_classic=True)
parser.set_defaults(use_nearestNeighbor=False)
parser.set_defaults(saveNets=False)

args = parser.parse_args()
use_lstm = args.use_lstm
numMinProp = args.propMin
numRandSys = args.systems
orbitType = args.orbit
if args.test is None:
    args.test = args.orbit
    args.testSys = numRandSys
elif args.test.isdigit():
    args.testSys = int(args.test)
    args.test = args.orbit
testSet = args.test
testSys = args.testSys
useOE = args.OE
useNoise = args.noise
useNorm = args.norm
useOnePass = args.one_pass
save_to_log = args.save_to_log
useEnergy=args.use_energy
useHybrid=args.use_hybrid
find_SW=args.find_SW
use_classic = args.use_classic
use_nearestNeighbor = args.use_nearestNeighbor
saveNets = args.saveNets

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import pandas as pd
from sklearn.metrics import log_loss, classification_report, confusion_matrix

from qutils.tictoc import timer
from qutils.ml.utils import getDevice, printModelParmSize
from qutils.ml.classifer import trainClassifier, LSTMClassifier, validateMultiClassClassifier
from qutils.ml.mamba import Mamba, MambaConfig, MambaClassifier
from qutils.ml.superweight import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight, findMambaSuperActivation,plotSuperActivation

strAdd = ""
if useEnergy:
    strAdd = strAdd + "Energy_"
if useOE:
    strAdd = strAdd + "OE_"
if useNorm:
    strAdd = strAdd + "Norm_"
if useNoise:
    strAdd = strAdd + "Noise_"
if useOnePass:
    strAdd = strAdd + "OnePass_"
if useHybrid:
    strAdd = strAdd + "Hybrid_"
# if use_classic:
#     strAdd = strAdd + "DT_"
if use_nearestNeighbor:
    strAdd = strAdd + "1-NN_"
if testSet != orbitType:
    strAdd = strAdd + "Test_" + testSet

logLoc = "gmat/data/classification/"+str(orbitType)+"/" + str(numMinProp) + "min-" + str(numRandSys) + "/"
logFileLoc = logLoc + str(numMinProp) + "min" + str(numRandSys)+ strAdd +'.log'
if save_to_log:
    import sys
    from contextlib import redirect_stdout, redirect_stderr
    import pandas as pd

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 10000)          # big number to avoid wrapping
    pd.set_option('display.expand_frame_repr', False)

    import warnings

    # Nuke everything (blunt):
    warnings.filterwarnings("ignore")

    # if location does not exist, create it
    import os
    if not os.path.exists("gmat/data/classification/"+str(orbitType)+"/" + str(numMinProp) + "min-" + str(numRandSys)):
        os.makedirs("gmat/data/classification/"+str(orbitType)+"/" + str(numMinProp) + "min-" + str(numRandSys))
    print("saving log output to {}".format(logFileLoc))

# display the data by calling the displayLogData.py script from its contained folder

class HybridClassifier(nn.Module):
    def __init__(self,config, input_size, hidden_size, num_layers, num_classes):
        super(HybridClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # Bidirectional LSTM
        )
        self.mamba = Mamba(config)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        """
        x: [batch_size, seq_length, input_size]
        """
        # h0, c0 default to zero if not provided
        out, (h_n, c_n) = self.lstm(x)
        h_n = self.mamba(out) # [batch_size, seq_length, hidden_size]

        # h_n is shape [num_layers, batch_size, hidden_size].
        # We typically take the last layer's hidden state: h_n[-1]
        last_hidden = h_n[:,-1,:]  # [batch_size, hidden_size]
        
        # Pass the last hidden state through a linear layer for classification
        logits = self.fc(last_hidden)  # [batch_size, num_classes]
        
        return logits

def main():
    import yaml
    with open("data.yaml", 'r') as f:
        dataConfig = yaml.safe_load(f)
    dataLoc = dataConfig['classification'] + orbitType +"/" + str(numMinProp) + "min-" + str(numRandSys)
    print(f"Processing datasets for {orbitType} with {numMinProp} minutes and {numRandSys} random systems.")
    # dataLoc = "c/Users/hu650776/GMAT-Thrust-Data/data/classification/data/classification/"+ orbitType +"/" + str(numMinProp) + "min-" + str(numRandSys)


    device = getDevice()

    batchSize = 16
    problemDim = 6

    # create a dictionary to hold yaml config values
    # TODO: change to pyyaml reading from a file 
    yaml_config = {}

    yaml_config['useOE'] = useOE
    yaml_config['useNorm'] = useNorm
    yaml_config['useNoise'] = useNoise
    yaml_config['useEnergy'] = useEnergy

    yaml_config['prop_time'] = numMinProp

    yaml_config['orbit'] = orbitType
    yaml_config['systems'] = numRandSys

    yaml_config['test_dataset'] = testSet
    yaml_config['test_systems'] = testSys

    from qutils.ml.classifer import prepareThrustClassificationDatasets

    train_loader, val_loader, test_loader, train_data,train_label,val_data,val_label,test_data,test_label = prepareThrustClassificationDatasets(yaml_config,dataConfig,output_np=True)

    # Hyperparameters
    input_size = train_data.shape[2] 
    hidden_factor = 8  # hidden size is a multiple of input size
    hidden_size = int(input_size * hidden_factor) # must be multiple of train dim
    num_layers = 1
    num_classes = 4  # e.g., multiclass classification
    learning_rate = 1e-3
    num_epochs = 100

    if useOnePass:
        num_epochs = 1

    criterion = torch.nn.CrossEntropyLoss()

    config = MambaConfig(d_model=input_size,n_layers = num_layers,expand_factor=hidden_size//input_size,d_state=32,d_conv=16,classifer=True)
    model_mamba = MambaClassifier(config,input_size, hidden_size, num_layers, num_classes).to(device).double()
    optimizer_mamba = torch.optim.Adam(model_mamba.parameters(), lr=learning_rate)

    schedulerPatience = 5

    scheduler_mamba = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_mamba,
        mode='min',             # or 'max' for accuracy
        factor=0.5,             # shrink LR by 50%
        patience=schedulerPatience             # wait for 3 epochs of no improvement
    )

    classlabels = ['No Thrust','Chemical','Electric','Impulsive']

    if useHybrid:
        config_hybrid = MambaConfig(d_model=hidden_size * 2,n_layers = 1,expand_factor=1,d_state=32,d_conv=16,classifer=True)

        model_hybrid = HybridClassifier(config_hybrid,input_size,hidden_size,num_layers,num_classes).to(device).double()
        optimizer_hybrid = torch.optim.Adam(model_hybrid.parameters(), lr=learning_rate)
        scheduler_hybrid = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_hybrid,
            mode='min',             # or 'max' for accuracy
            factor=0.5,             # shrink LR by 50%
            patience=schedulerPatience
        )

        print('\nEntering Hybrid Training Loop')
        trainClassifier(model_hybrid,optimizer_hybrid,scheduler_hybrid,[train_loader,test_loader,val_loader],criterion,num_epochs,device,classLabels=classlabels)
        printModelParmSize(model_hybrid)

        if testSet != orbitType:
            validateMultiClassClassifier(model_hybrid,test_loader,criterion,num_classes,device,classlabels,printReport=True)
        else:
            validateMultiClassClassifier(model_hybrid,val_loader,criterion,num_classes,device,classlabels,printReport=True)

    if use_classic:
        from lightgbm import LGBMClassifier

        print("\nEntering Decision Trees Training Loop")
        DTTimer = timer()
        def printClassicModelSize(model):
            import tempfile, pathlib

            with tempfile.TemporaryDirectory() as tmp:
                path = pathlib.Path(tmp) / "model.bin"   # any extension is fine
                model.booster_.save_model(str(path))     # binary dump by default
                size_bytes = path.stat().st_size
            print("\n==========================================================================================")
            print(f"Total parameters: NaN")
            print(f"Total memory (bytes): {size_bytes}")
            print(f"Total memory (MB): {size_bytes / (1024 ** 2)}")
            print("==========================================================================================")

        def validate_lightgbm(model, val_loader, num_classes, classlabels=None, print_report=True):
            """Evaluate a trained LightGBM multiclass classifier on a PyTorch‑style DataLoader.

            * model          - fitted lightgbm.LGBMClassifier (objective='multiclass')
            * val_loader     - yields (seq_batch, label_batch); seq_batch can be torch.Tensor or np.ndarray
                            Shape per sample must match training: (7, L).  Flatten before predict.
            * num_classes    - integer (4 in your case)
            """
            # --------------------------------------------------------------------- #
            # Aggregate validation data                                             #
            # --------------------------------------------------------------------- #
            X_list, y_list = [], []
            for seq, lab in val_loader:
                # → ndarray, shape (batch, 7*L)
                xb = (seq if isinstance(seq, np.ndarray) else seq.cpu().numpy()).reshape(seq.shape[0], -1)
                yb = (lab if isinstance(lab, np.ndarray) else lab.cpu().numpy())
                X_list.append(xb)
                y_list.append(yb)

            X_val = np.concatenate(X_list, axis=0)
            y_true = np.concatenate(y_list, axis=0)

            # --------------------------------------------------------------------- #
            # Predict                                                               #
            # --------------------------------------------------------------------- #
            proba = model.predict_proba(X_val, num_iteration=model.best_iteration_)
            y_pred = proba.argmax(axis=1)

            # --------------------------------------------------------------------- #
            # Metrics                                                               #
            # --------------------------------------------------------------------- #
            val_loss = log_loss(y_true, proba, labels=np.arange(num_classes))
            accuracy = 100.0 * (y_pred == y_true).mean()

            # Per‑class accuracy
            class_tot = np.bincount(y_true, minlength=num_classes)
            class_corr = np.bincount(y_true[y_true == y_pred], minlength=num_classes)
            per_class_acc = 100.0 * class_corr / np.maximum(class_tot, 1)

            # --------------------------------------------------------------------- #
            # Reporting                                                             #
            # --------------------------------------------------------------------- #
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%\n")

            print("Per-Class Validation Accuracy:")
            for i in range(num_classes):
                label = classlabels[i] if classlabels else f"Class {i}"
                if class_tot[i]:
                    print(f"  {label}: {per_class_acc[i]:.2f}% ({class_corr[i]}/{class_tot[i]})")
                else:
                    print(f"  {label}: No samples")

            if print_report:
                print("\nClassification Report:")
                print(
                    classification_report(
                        y_true, y_pred,
                        labels=list(range(num_classes)),
                        target_names=(classlabels if classlabels else None),
                        digits=4,
                        zero_division=0,
                    )
                )

                cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
                print("\nConfusion Matrix (rows = true, cols = predicted):")
                print(
                    pd.DataFrame(
                        cm,
                        index=[f"T_{cls}" for cls in (classlabels if classlabels else range(num_classes))],
                        columns=[f"P_{cls}" for cls in (classlabels if classlabels else range(num_classes))]
                    )
                )

            return val_loss, accuracy
        classicModel = LGBMClassifier(objective="multiclass",num_classes=num_classes,n_estimators=4,max_depth=-1,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,verbosity=-1)   # or 'verbose' for older builds)
        
        # flatten features
        X_train = train_data.reshape(train_data.shape[0], -1).astype(np.float32)    # (number of systems to train on, network features * length of time series)    
        y_train = train_label.reshape(-1).astype(np.int32)             # (number of systems to train on,)
        classicModel.fit(X_train, y_train)
        DTTimer.toc()
        printClassicModelSize(classicModel)
        print("\nDecision Trees Validation")
        DTTimerInference = timer()
        if testSet != orbitType:
            validate_lightgbm(classicModel, test_loader, num_classes, classlabels=classlabels, print_report=True)
        else:
            validate_lightgbm(classicModel, val_loader, num_classes, classlabels=classlabels, print_report=True)
        DTTimerInference.tocStr("Decision Trees Inference Time")
    if use_nearestNeighbor:
        def z_normalize(ts, eps=1e-8):
            # ts: [T] or [T,C]
            mean = ts.mean(axis=0, keepdims=True)
            std = ts.std(axis=0, keepdims=True)
            return (ts - mean) / (std + eps)

        def train_data_z_normalize(train_data):
            """Z-normalize training data along the time axis."""
            return np.array([z_normalize(ts) for ts in train_data])

        def print1_NNModelSize(model):
            import tempfile, pathlib
            import pickle

            with tempfile.TemporaryDirectory() as tmp:
                path = pathlib.Path(tmp) / "model.pkl"
                with open(path, "wb") as f:
                    pickle.dump(model, f)
                size_bytes = path.stat().st_size

            print("\n" + "=" * 90)
            print(f"Total parameters: NaN (non-parametric model)")
            print(f"Total memory (bytes): {size_bytes}")
            print(f"Total memory (MB): {size_bytes / (1024 ** 2):.4f}")
            print("=" * 90)

        def validate_1NN(clf, val_loader, num_classes, classlabels=None):
            """Evaluate a 1-NN classifier (e.g., sktime KNeighborsTimeSeriesClassifier) on a PyTorch DataLoader."""
            X_val_list, y_val_list = [], []

            for seq, lab in val_loader:
                xb = seq.cpu().numpy()  # preserve time-series shape
                yb = lab.cpu().numpy()
                X_val_list.append(xb) #z-normalize each time series
                y_val_list.append(yb)

            # Merge batches
            X_val_np = np.concatenate(X_val_list, axis=0)
            y_true = np.concatenate(y_val_list)

            # Adapt shape for sktime: [N,C,T]
            # [N,T,C] → [N,C,T]
            X_val_np = np.transpose(X_val_np, (0, 2, 1))

            # Predict
            y_pred = clf.predict(X_val_np)

            # Accuracy
            correct = (y_pred == y_true).sum()
            total = len(y_true)
            accuracy = 100.0 * correct / total

            print(f"Validation Loss: NaN, Validation Accuracy: {accuracy:.2f}%\n")

            # Per-class accuracy
            class_corr = np.zeros(num_classes, dtype=int)
            class_tot = np.zeros(num_classes, dtype=int)
            for yt, yp in zip(y_true, y_pred):
                class_tot[yt] += 1
                if yt == yp:
                    class_corr[yt] += 1
            per_class_acc = 100.0 * class_corr / np.maximum(class_tot, 1)

            print("Per-Class Validation Accuracy:")
            for i in range(num_classes):
                label = classlabels[i] if classlabels else f"Class {i}"
                if class_tot[i]:
                    print(f"  {label}: {per_class_acc[i]:.2f}% ({class_corr[i]}/{class_tot[i]})")
                else:
                    print(f"  {label}: No samples")

            print("\nClassification Report:")
            print(
                classification_report(
                    y_true, y_pred,
                    labels=list(range(num_classes)),
                    target_names=(classlabels if classlabels else None),
                    digits=4,
                    zero_division=0,
                )
            )

            cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
            print("\nConfusion Matrix (rows = true, cols = predicted):")
            print(
                pd.DataFrame(
                    cm,
                    index=[f"T_{cls}" for cls in (classlabels if classlabels else range(num_classes))],
                    columns=[f"P_{cls}" for cls in (classlabels if classlabels else range(num_classes))]
                )
            )

        from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

        print("\nEntering Nearest Neighbor Training Loop")
        dtw = timer()
        # [N,T,C] -> [N,C,T]
        train_data_NN = np.transpose(train_data, (0, 2, 1))

        # train_data_NN = train_data_z_normalize(train_data_NN)  # Z-normalize along time axis

        clf = KNeighborsTimeSeriesClassifier(
            n_neighbors=1,
            distance="dtw",
            distance_params={"sakoe_chiba_radius": 10}
    )
        clf.fit(train_data_NN, train_label)
        dtw.toc()
        print1_NNModelSize(clf)
        print("\n1-NN Validation")
        dtwInference = timer()
        if testSet != orbitType:
            validate_1NN(clf, test_loader, num_classes, classlabels=classlabels)
        else:
            validate_1NN(clf, val_loader, num_classes, classlabels=classlabels)
        dtwInference.tocStr("1-NN Inference Time")

    if use_lstm:
        model_LSTM = LSTMClassifier(input_size, int(3*hidden_size//4), num_layers, num_classes).to(device).double()
        optimizer_LSTM = torch.optim.Adam(model_LSTM.parameters(), lr=learning_rate)
        scheduler_LSTM = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_LSTM,
            mode='min',             # or 'max' for accuracy
            factor=0.5,             # shrink LR by 50%
            patience=schedulerPatience
        )

        print('\nEntering LSTM Training Loop')
        trainClassifier(model_LSTM,optimizer_LSTM,scheduler_LSTM,[train_loader,test_loader,val_loader],criterion,num_epochs,device,classLabels=classlabels)
        printModelParmSize(model_LSTM)
        print("\nLSTM Validation")
        LSTMInference = timer()
        if testSet != orbitType:
            validateMultiClassClassifier(model_LSTM,test_loader,criterion,num_classes,device,classlabels,printReport=True)
        else:
            validateMultiClassClassifier(model_LSTM,val_loader,criterion,num_classes,device,classlabels,printReport=True)
        LSTMInference.tocStr("LSTM Inference Time")

    print('\nEntering Mamba Training Loop')
    trainClassifier(model_mamba,optimizer_mamba,scheduler_mamba,[train_loader,test_loader,val_loader],criterion,num_epochs,device,classLabels=classlabels)
    printModelParmSize(model_mamba)
    print("\nMamba Validation")
    mambaInference = timer()
    if testSet != orbitType:
        validateMultiClassClassifier(model_mamba,test_loader,criterion,num_classes,device,classlabels,printReport=True)
    else:
        validateMultiClassClassifier(model_mamba,val_loader,criterion,num_classes,device,classlabels,printReport=True)
    mambaInference.tocStr("Mamba Inference Time")

    if saveNets:
        import os
        if not os.path.exists("gmat/data/classification/"+str(orbitType)+"/" + str(numMinProp) + "min-" + str(numRandSys)):
            os.makedirs("gmat/data/classification/"+str(orbitType)+"/" + str(numMinProp) + "min-" + str(numRandSys))
        print(f"Saving networks to gmat/data/classification/{orbitType}/{numMinProp}min-{numRandSys}/")
        if use_lstm:
            torch.save(model_LSTM.state_dict(), f"{logLoc}lstm_"+ orbitType +"_"+strAdd+".pt")
        if useHybrid:
            torch.save(model_hybrid.state_dict(), f"{logLoc}hybrid_"+ orbitType +"_"+strAdd+".pt")
        torch.save(model_mamba.state_dict(), f"{logLoc}mamba_"+ orbitType +"_"+strAdd+".pt")

    if find_SW:
        magnitude, index = findMambaSuperActivation(model_mamba,torch.tensor(test_data).to(device))
        # super activation returns the entire mamba network parameters, but the classifier does not use the out_proj layer
        # so we drop it
        magnitude = magnitude[:-1]
        index = index[:-1]
        # also drop the x_proj layer, no longer needed as well
        magnitude.pop(2)
        index.pop(2)


        normedMagsMRP = np.zeros((len(magnitude),))
        for i in range(len(magnitude)):
            normedMagsMRP[i] = magnitude[i].norm().detach().cpu()

        printoutMaxLayerWeight(model_mamba)
        getSuperWeight(model_mamba)
        plotSuperWeight(model_mamba)
        plotSuperActivation(magnitude, index,printOutValues=True,mambaLayerAttributes = ["in_proj","conv1d","dt_proj"])
        plt.title("Mamba Classifier Super Activations")


# # example onnx export
# # # generate example inputs for ONNX export
# example_inputs = torch.randn(1, numMinProp, input_size).to(device).double()
# # export the model to ONNX format
# # Note: `dynamo=True` is used to enable PyTorch's dynamo for better performance and compatibility.
# onnx_path = f"{dataLoc}/mambaTimeSeriesClassificationGMATThrusts.onnx"
# onnx_program = torch.onnx.export(model_mamba, example_inputs,onnx_path)
# print(f"ONNX model saved to {onnx_path}")

if __name__ == "__main__":
    if save_to_log:
        log = logFileLoc  # path
        with open(log, 'w', buffering=1, encoding='utf-8') as f, \
            redirect_stdout(f), redirect_stderr(f):
            main()
    else:
        main()
    plt.show()