import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score

from qutils.mamba import Mamba, MambaConfig
from qutils.ml import getDevice

forceConst = 10.0 

# Parameters
num_features = 2     # toy example, 2 features
num_classes = 1    # binary classification
hiddenSize = 16

sampling_rate = 100  # Hz
duration = 10        # seconds
seq_len = sampling_rate * duration

t = np.linspace(0, duration, seq_len)

device = getDevice()

# Generate synthetic data
numRandSys = 1000
def generate_batch_forced_oscillators(num_systems=numRandSys):
    dt = t[1] - t[0]

    omega = 2 * np.pi * 1.0  # natural frequency
    zeta = 0.1               # damping

    features_all = np.zeros((num_systems, seq_len, 2), dtype=np.float32)
    labels_all = np.zeros((num_systems, seq_len), dtype=np.float32)

    for n in range(num_systems):
        x = np.zeros(seq_len)
        v = np.zeros(seq_len)
        a = np.zeros(seq_len)

        # Random initial conditions
        x[0] = np.random.rand()
        v[0] = np.random.rand()

        # Random force magnitude in [0, 5]
        F0 = forceConst * np.random.rand()

        # Random force duration in (0, 1] seconds
        force_duration = np.random.uniform(low=0.1, high=1.0)

        # Random start time such that force doesn't go past simulation end
        max_start_time = duration - force_duration
        force_start_time = np.random.uniform(low=0.0, high=max_start_time)
        force_end_time = force_start_time + force_duration

        # External force signal
        f = np.zeros(seq_len)
        f[(t >= force_start_time) & (t < force_end_time)] = F0

        # Integrate using Euler
        for i in range(1, seq_len):
            a[i-1] = f[i-1] - 2*zeta*omega*v[i-1] - omega**2 * x[i-1]
            v[i] = v[i-1] + a[i-1] * dt
            x[i] = x[i-1] + v[i-1] * dt

        # Store features: [x, v]
        features_all[n, :, 0] = x
        features_all[n, :, 1] = v

        # Store label = 1 during force application
        labels_all[n, (t >= force_start_time) & (t < force_end_time)] = 1.0

    return (
        torch.tensor(features_all, dtype=torch.float32),  # shape [N, T, 2]
        torch.tensor(labels_all, dtype=torch.float32)     # shape [N, T]
    )

# Define the model
class LSTMSequenceClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: [batch_size, seq_length, input_size]
        """
        # h0, c0 default to zero if not provided
        out, (h_n, c_n) = self.lstm(x)
        
        # h_n is shape [num_layers, batch_size, hidden_size].
        # We typically take the last layer's hidden state: h_n[-1]
        
        # Pass the last hidden state through a linear layer for classification on a per timestep basis
        logits = self.classifier(out)       # logits: [B, T, 1]
        logits = logits.squeeze(-1)         # logits: [B, T]
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
        
        # Pass the last hidden state through a linear layer for classification on a per timestep basis
        logits = self.fc(h_n)               # logits: [B, T, 1]
        logits = logits.squeeze(-1)         # logits: [B, T]

        return logits

# Training loop
def train_model(model, train_loader, val_loader, num_epochs=10):
    # uses 3 new metrics
    # 1. Precision
    # true positives / (true positives + false positives)
    # >= 0.8 -- few false alarms
    # 2. Recall
    # true positives / (true positives + false negatives)
    # >= 0.8 -- rarely misses true events
    # 3. F1 Score
    # 2 * (precision * recall) / (precision + recall)
    # >= 0.8 -- well balanced detection

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    schedulerPatience = 3
    best_loss = float('inf')
    ESpatience = schedulerPatience * 2
    counter = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=schedulerPatience,
        verbose=True
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            logits = model(sequences)               # [B, T]
            loss = criterion(logits, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}")

        # -------------------
        # Validation Metrics
        # -------------------
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)

                logits = model(sequences)               # [B, T]
                probs = torch.sigmoid(logits)           # [B, T]
                preds = (probs >= 0.5).int()             # [B, T]

                all_preds.append(preds.cpu().flatten())
                all_targets.append(labels.cpu().flatten().int())

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)

        print(f"Validation Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")

        # -------------------
        # Validation Loss
        # -------------------
        val_loss = 0.0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device).float()
                logits = model(sequences)
                loss = criterion(logits, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0
            # Optionally save the best model
        else:
            counter += 1
            if counter >= ESpatience:
                print("Early stopping")
                break
# Evaluation
def evaluate_model(model, loader, sample_batch_idx=6, sample_in_batch_idx=6):
    """
    Visualizes per-timestep classification for a sample from a DataLoader.
    
    Parameters:
        model: trained model
        loader: DataLoader (e.g., val_loader)
        sample_batch_idx: index of the batch to sample from
        sample_in_batch_idx: index within the selected batch
    """
    model.eval()
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(loader):
            if batch_idx == sample_batch_idx:
                x_sample = batch_x[sample_in_batch_idx].unsqueeze(0).to(device)  # [1, T, D]
                y_sample = batch_y[sample_in_batch_idx].to(device)               # [T]
                break
        else:
            raise ValueError("Sample batch index out of range.")

        logits = model(x_sample).squeeze(0)           # [T]
        probs = torch.sigmoid(logits)                 # [T]
        predictions = (probs > 0.5).long()            # [T]

        plt.figure(figsize=(12, 4))
        plt.plot(t,y_sample.cpu().numpy(), label='True Label', linewidth=2)
        plt.plot(t,predictions.cpu().numpy(), linestyle='--', label='Predicted Label', linewidth=1)
        plt.legend()
        plt.title(f"Per-Timestep Classification (Batch {sample_batch_idx}, Sample {sample_in_batch_idx})")
        plt.xlabel("Time Step")
        plt.ylabel("Label")
        plt.grid(True)
        plt.tight_layout()

# Run the pipeline
x, y = generate_batch_forced_oscillators()

batchSize = 16

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15


total_samples = len(x)
train_end = int(train_ratio * total_samples)
val_end = int((train_ratio + val_ratio) * total_samples)

# Split the data
train_data = x[:train_end]
train_label = y[:train_end]

val_data = x[train_end:val_end]
val_label = y[train_end:val_end]

test_data = x[val_end:]
test_label = y[val_end:]

train_dataset = TensorDataset(train_data,train_label)
val_dataset = TensorDataset(val_data,val_label)
test_dataset = TensorDataset(test_data,test_label)

train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

model = LSTMSequenceClassifier(input_dim=num_features, hidden_dim=hiddenSize, num_layers=1, num_classes=num_classes).to(device)
config = MambaConfig(d_model=num_features,n_layers = 1,expand_factor=hiddenSize//num_features,d_state=hiddenSize,d_conv=16,classifer=True)
mambaModel = MambaClassifier(config, input_size=num_features, hidden_size=hiddenSize, num_layers=1, num_classes=num_classes).to(device)

train_model(model, train_loader,val_loader, num_epochs=100)
evaluate_model(model, val_loader)
plt.gcf()
plt.title("LSTM Model Evaluation on Random Sample")

print("\n")

train_model(mambaModel, train_loader,val_loader, num_epochs=100)
evaluate_model(mambaModel, val_loader)
plt.gcf()
plt.title("Mamba Model Evaluation on Random Sample")

plt.show()
