import numpy as np
import matplotlib.pyplot as plt
import torch
import control as ct
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from qutils.tictoc import timer

from qutils.ml import printModelParmSize, getDevice
from qutils.mamba import Mamba, MambaConfig
from qutils.integrators import ode87

#set webagg backend for matplotlib - i've been liking it 
# plt.switch_backend('WebAgg')

import sys

plotOn = True
seqLength = 100
numRandSys = 1000
deltaVConst = 1.0 # m/s

import torch.nn as nn
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """
        x: [batch_size, seq_length, input_size]
        """
        # h0, c0 default to zero if not provided
        out, (h_n, c_n) = self.lstm(x)
        
        # h_n is shape [num_layers, batch_size, hidden_size].
        # We typically take the last layer's hidden state: h_n[-1]
        last_hidden = h_n[-1]  # [batch_size, hidden_size]
        
        # Pass the last hidden state through a linear layer for classification
        logits = self.fc(last_hidden)  # [batch_size, num_classes]
        
        return logits
        
#tranformer classifier for time series data
class TransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(TransformerClassifier, self).__init__()
        
        self.d_model = hidden_size  # Output of transformer & input to fc
        self.embedding = nn.Linear(input_size, self.d_model)  # Project input to match d_model

        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=8,  # Make sure d_model % nhead == 0
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=64,  # Internal feedforward layer size inside Transformer
            batch_first=True
        )
        
        self.fc = nn.Linear(self.d_model, num_classes)  # Final classification layer

    def forward(self, x):
        """
        x: [batch_size, seq_length, input_size]
        """
        x = self.embedding(x)         # [batch_size, seq_length, d_model]
        out = self.transformer(x, x)  # [batch_size, seq_length, d_model]
        last_output = out[:, -1, :]   # [batch_size, d_model]
        logits = self.fc(last_output) # [batch_size, num_classes]
        return logits
    
class MambaClassifier(nn.Module):
    def __init__(self,config, input_size, hidden_size, num_layers, num_classes):
        super(MambaClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.mamba = Mamba(config)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """
        x: [batch_size, seq_length, input_size]
        """
        
        h_n = self.mamba(x)
        
        # h_n is shape [batch_size, seq_length, hidden_size].
        # We typically take the last layer's hidden state: h_n[:,-1,:]
        last_hidden = h_n[:,-1,:]  # [batch_size, hidden_size]
        
        # Pass the last hidden state through a linear layer for classification
        logits = self.fc(last_hidden)  # [batch_size, num_classes]
        
        return logits


rng = np.random.default_rng(seed=1) # Seed for reproducibility

device = getDevice()

batchSize = 16
problemDim = 2

# Hyperparameters
input_size = problemDim   # 2
hidden_size = 64
num_layers = 1
num_classes = 1  # e.g., binary classification
learning_rate = 1e-2
num_epochs = 100

config = MambaConfig(d_model=input_size,n_layers = num_layers,expand_factor=hidden_size//input_size,d_state=32,d_conv=16,classifer=True)


numericalResultForced = np.zeros((numRandSys,int(seqLength*2),problemDim))
numericalResultUnforced = np.zeros((numRandSys,int(seqLength*2),problemDim))

mu = 3.986004418e14  # Earth’s mu in m^3/s^2
R = 6371e3 # radius of earth in m
dt = 1.0 # time step in seconds
t0 = 0.0 # start time in seconds


def twoBody(t, y, p=mu):
    r = y[0:2]
    R = np.linalg.norm(r)

    dydt1 = y[2]
    dydt2 = y[3]

    dydt3 = -p / R**3 * y[0]
    dydt4 = -p / R**3 * y[1]

    return np.array([dydt1, dydt2,dydt3,dydt4])


timeToGenData = timer()
for i in range(numRandSys):
    # generate planar cicular orbits in leo with random parameters
    r1 = rng.uniform(R + 200e3,R + 1000e3) # radius in m
    v1 = np.sqrt(mu / r1)
    # Let's align it so we start at (r1,0) and velocity in the +y direction
    y0 = np.array([r1, 0.0, 0.0, v1])
    tf = 2*np.pi*np.sqrt(r1**3/mu) # time to complete one orbit in seconds

    t1_beforeBurn = rng.uniform(0,tf/2) # time before first burn
    t_beforeBurn = np.linspace(t0,t1_beforeBurn,seqLength)
    # propagate the system unforced
    tODE_beforeBurn,yODE_beforeBurn = ode87(fun=lambda t, y: twoBody(t, y, mu), tspan=(t0, t1_beforeBurn), y0=y0, t_eval = t_beforeBurn,rtol=1e-8, atol=1e-10)

    y_burn1 = yODE_beforeBurn[-1,:]   # [x, y, vx, vy]


    # generate random delta v for hohmann transfer
    deltaV = rng.uniform(0,deltaVConst) # delta v in m/s
    # get unit vector in direction of velocity
    v = np.array([y_burn1[2], y_burn1[3]])  # (vx, vy)
    v = v/np.linalg.norm(v) * deltaV  # Normalize to get unit vector and apply delta-v in that direction
    dvx, dvy = v[0], v[1]

    y_burn1[2] += dvx
    y_burn1[3] += dvy

    t2_Burn = rng.uniform(t1_beforeBurn,tf) # time after burn
    t_Burn = np.linspace(t1_beforeBurn,t2_Burn,seqLength)
    # propagate the system (forced) based on the transfer orbit calculated
    tODE_Burn,yODE_Burn = ode87(fun=lambda t, y: twoBody(t, y, mu), tspan=(t1_beforeBurn, t2_Burn), y0=y_burn1, t_eval = t_Burn,rtol=1e-8, atol=1e-10)

    tOriginalCircOrbit,yOriginalCircOrbit = ode87(fun=lambda t, y: twoBody(t, y, mu),tspan=(0, t2_Burn),y0=y0,t_eval = np.linspace(0,t2_Burn,int(seqLength*2)), rtol=1e-8, atol=1e-10)
    tOriginalCircOrbitPlot,yOriginalCircOrbitPlot = ode87(fun=lambda t, y: twoBody(t, y, mu),tspan=(0, tf),y0=y0,t_eval = np.linspace(0,tf,int(seqLength*2)), rtol=1e-8, atol=1e-10)


    # concatenate the two trajectories
    yODE = np.concatenate((yODE_beforeBurn,yODE_Burn),axis=0)
    # concatenate the time vectors
    t = np.concatenate((tODE_beforeBurn,tODE_Burn),axis=0)

   
    numericalResultForced[i,:,:] = yODE[:,:2]
    numericalResultUnforced[i,:,:] = yOriginalCircOrbit[:,:2]

print("Time to generate data: {:.2f} seconds".format(timeToGenData.tocVal()))

plt.figure(figsize=(10, 10))
plt.title('Hohmann Transfer Orbit')
plt.plot(yOriginalCircOrbitPlot[:,0], yOriginalCircOrbitPlot[:,1], 'k--', label='Original Circular Orbit')
plt.plot(yOriginalCircOrbit[:,0], yOriginalCircOrbit[:,1], 'g--', label='Unforced Orbit')
plt.plot(yODE_beforeBurn[:,0], yODE_beforeBurn[:,1], 'b-', label='Before Burn')
plt.plot(yODE_Burn[:,0], yODE_Burn[:,1], 'r-', label='After Burn')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()  # Show the plot of the trajectory

# make data labels for a two class problem of shape numRandSys x 2, 1, 0 for forced and unforced respectively
ForcedLabel = np.ones(numRandSys)
UnforcedLabel = np.zeros(numRandSys)

dataset = np.concatenate((numericalResultForced,numericalResultUnforced),axis=0)
dataset_label = np.concatenate((ForcedLabel,UnforcedLabel),axis=0).reshape(2*numRandSys,1)

indices = np.random.permutation(dataset.shape[0])

dataset = dataset[indices]
dataset_label = dataset_label[indices]

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15


total_samples = len(dataset)
train_end = int(train_ratio * total_samples)
val_end = int((train_ratio + val_ratio) * total_samples)

# Split the data
train_data = dataset[:train_end]
train_label = dataset_label[:train_end]

val_data = dataset[train_end:val_end]
val_label = dataset_label[train_end:val_end]

test_data = dataset[val_end:]
test_label = dataset_label[val_end:]

train_dataset = TensorDataset(torch.from_numpy(train_data).float(),torch.from_numpy(train_label).long())
val_dataset = TensorDataset(torch.from_numpy(val_data).float(),torch.from_numpy(val_label).long())
test_dataset = TensorDataset(torch.from_numpy(test_data).float(),torch.from_numpy(test_label).long())

train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

model_mamba = MambaClassifier(config,input_size, hidden_size, num_layers, num_classes).to(device).float()
model_LSTM = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device).float()
model_transformer = TransformerClassifier(input_size, hidden_size, num_layers, num_classes).to(device).float()

criterion = nn.BCEWithLogitsLoss() # for binary classification
optimizer_mamba = torch.optim.Adam(model_mamba.parameters(), lr=learning_rate)
optimizer_LSTM = torch.optim.Adam(model_LSTM.parameters(), lr=learning_rate)
optimizer_transformer = torch.optim.Adam(model_transformer.parameters(), lr=learning_rate)

schedulerPatience = 3
scheduler_mamba = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_mamba,
    mode='min',             # or 'max' for accuracy
    factor=0.5,             # shrink LR by 50%
    patience=schedulerPatience,             # wait for 3 epochs of no improvement
    verbose=True
)

scheduler_LSTM = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_LSTM,
    mode='min',             # or 'max' for accuracy
    factor=0.5,             # shrink LR by 50%
    patience=schedulerPatience,             # wait for 3 epochs of no improvement
    verbose=True
)


scheduler_transformer = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_transformer,
    mode='min',             # or 'max' for accuracy
    factor=0.5,             # shrink LR by 50%
    patience=schedulerPatience,             # wait for 3 epochs of no improvement
    verbose=True
)


# Training loop
def trainClassifier(model,optimizer,scheduler):
    best_loss = float('inf')
    ESpatience = schedulerPatience * 2  # patience for early stopping
    counter = 0
    # early stopping by user control ctrl+c to break the training loop
    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0

            for sequences, labels in train_loader:
                # Move data to GPU
                sequences = sequences.to(device)  # [batch_size, seq_length, input_size]
                labels = labels.to(device)        # [batch_size]
                
                # Forward pass
                logits = model(sequences)        # [batch_size, num_classes]
                loss = criterion(logits, labels.float())  # [batch_size]
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

            # Validation (optional quick check)
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences = sequences.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(sequences)
                    probs = torch.sigmoid(outputs)                  # Convert logits to probabilities
                    predicted = (probs >= 0.5).float()              # Threshold at 0.5
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100.0 * correct / total
            print(f"Validation Accuracy: {accuracy:.2f}%")

            val_loss = 0.0
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences = sequences.to(device)
                    labels = labels.to(device).float()
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)

            scheduler.step(avg_val_loss)  # <-- update LR based on validation loss
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                counter = 0  # reset
                # save best model if desired
            else:
                counter += 1
                if counter >= ESpatience:
                    print("Early stopping")
                    break

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        return

print('\nEntering LSTM Training Loop')
LSTMTrainTime = timer()
trainClassifier(model_LSTM,optimizer_LSTM,scheduler_LSTM)
LSTMTrainTime.toc()
printModelParmSize(model_LSTM)

print('\nEntering Mamba Training Loop')
mambaTrainTime = timer()
trainClassifier(model_mamba,optimizer_mamba,scheduler_mamba)
mambaTrainTime.toc()
printModelParmSize(model_mamba)

print('\nEntering Transformer Training Loop')
transformerTrainTime = timer()
trainClassifier(model_transformer,optimizer_transformer,scheduler_transformer)
transformerTrainTime.toc()
printModelParmSize(model_transformer)



if plotOn:
    plt.show()
 