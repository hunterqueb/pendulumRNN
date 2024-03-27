import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def create_dataset(dataset,device,lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction

    for how big the lookback window is, the number of points is removed from the data sets, need to fix!

    there are a few ways to get around this,
    1. padding the data
    2. using partial windows

    padding the data seems like a bad approach since in regresssion task, there are no limits for the specific value
    that a dynamical system could take. this could bring issue when generalizing this approach to other systems,
    introduce artifacts into the system, and generally cause hallucinations

    partial windows seem like the best approach with conserving the integrity of the output data
        need to examine this more later

        


        
        import torch

        # Example dataset
        data = torch.randn(100, 1)  # Replace with your dataset (100 data points, 1 feature per point)

        lookback_window = 2
        processed_data = []

        for i in range(len(data)):
            start_idx = max(0, i - lookback_window)
            window = data[start_idx:i+1]
            processed_data.append(window)

        # Now, processed_data is a list of tensors of varying lengths

        from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

        # Sort your batch by sequence length in descending order
        processed_data.sort(key=lambda x: len(x), reverse=True)

        # Pack the sequences
        packed_input = pack_sequence(processed_data)

        # Process with LSTM
        lstm_out, _ = lstm(packed_input)

        # Optionally, unpack the sequences
        unpacked, lengths = pad_packed_sequence(lstm_out)


    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    X = np.array(X)
    y = np.array(y)

    return torch.tensor(X).double().to(device), torch.tensor(y).double().to(device)


def generateTrajectoryPrediction(train_plot,test_plot):
    # takes matrices of two equal lengths and compares the values element by element. 
    # if a number occupys one matrix but not the other return a new matrix with the nonzero value.
    # if both matrices have nan, a new matrix is returned with the nan value.
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

class SelfAttentionLayer(nn.Module):
   def __init__(self, feature_size):
       super(SelfAttentionLayer, self).__init__()
       self.feature_size = feature_size

       # Linear transformations for Q, K, V from the same source
       self.key = nn.Linear(feature_size, feature_size)
       self.query = nn.Linear(feature_size, feature_size)
       self.value = nn.Linear(feature_size, feature_size)

   def forward(self, x, mask=None):
       # Apply linear transformations
       keys = self.key(x)
       queries = self.query(x)
       values = self.value(x)

       # Scaled dot-product attention
       scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

       # Apply mask (if provided)
       if mask is not None:
           scores = scores.masked_fill(mask == 0, -1e9)

       # Apply softmax
       attention_weights = F.softmax(scores, dim=-1)

       # Multiply weights with values
       output = torch.matmul(attention_weights, values)

       return output, attention_weights

class LSTMSelfAttentionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_value, heads=1):
        super(LSTMSelfAttentionNetwork, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,dropout=dropout_value)

        # Self-attention layer
        self.self_attention = SelfAttentionLayer(hidden_dim)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Pass data through LSTM layer
        lstm_out, lstm_hidden = self.lstm(x)

        # Pass data through self-attention layer
        attention_out, attention_weights = self.self_attention(lstm_out,mask=None)

        # Pass data through fully connected layer
        final_out = self.fc(attention_out)

        return final_out

class LSTMSelfAttentionNetwork2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_value, heads=1):
        super(LSTMSelfAttentionNetwork2, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,dropout=dropout_value)

        # Self-attention layer
        self.self_attention = SelfAttentionLayer(hidden_dim)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.fcOut = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Pass data through LSTM layer
        lstm_out, lstm_hidden = self.lstm(x)

        # Pass data through self-attention layer
        attention_out, attention_weights = self.self_attention(lstm_out,mask=None)

        # Pass data through fully connected layer
        fcout = self.fc(attention_out)
        final_out = self.fcOut(fcout)

        return final_out

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_value, heads=1):
        super(LSTM, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,dropout=dropout_value)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Pass data through LSTM layer
        lstm_out, _ = self.lstm(x)

        # Pass data through fully connected layer
        final_out = self.fc(lstm_out)

        return final_out

def transferLSTM(pretrainedModel,newModel,trainableLayer = [True, True, True]):
    '''
    custom function to transfer knowledge of LSTM network from a pretrained model to a new model

    parameters: pretrainedModel - pretrained pytorch model with one LSTM layer and one self attention layer
                newModel - untrained pytorch model with one LSTM layer and one self attention layer
    '''
    newModel.lstm.load_state_dict(pretrainedModel.lstm.state_dict())
    newModel.self_attention.load_state_dict(pretrainedModel.self_attention.state_dict())
    newModel.fc.load_state_dict(pretrainedModel.fc.state_dict())

    # Freeze the weights of the LSTM layers
    for param in newModel.lstm.parameters():
        param.requires_grad = trainableLayer[0]
    for param in newModel.self_attention.parameters():
        param.requires_grad = trainableLayer[1]
    for param in newModel.fc.parameters():
        param.requires_grad = trainableLayer[2]
    return newModel


class CNN_LSTM_SA(nn.Module):
    def __init__(self):
        super(CNN_LSTM_SA, self).__init__()
        # CNN part
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # LSTM part
        self.lstm = nn.LSTM(32, 50, batch_first=True) # 32 features from CNN, 50 hidden units

        # Self-attention layer
        self.self_attention = SelfAttentionLayer(50)

        # Regression output
        self.fc = nn.Linear(50, 1)  # Assuming a single continuous value as output

    def forward(self, x):
        # x shape: [batch, seq_len, channels, height, width]
        batch_size, seq_len, C, H, W = x.size()
        
        # Reshape for CNN
        c_in = x.view(batch_size * seq_len, C, H, W)
        c_out = F.relu(self.conv1(c_in))
        c_out = self.pool(F.relu(self.conv2(c_out)))
        
        # Reshape for LSTM
        r_out = c_out.view(batch_size, seq_len, -1)

        # LSTM output
        lstm_out, _ = self.lstm(r_out)
        lstm_out = lstm_out[:, -1, :]  # Get the output of the last time step

        attention_out, attention_weights = self.self_attention(lstm_out,mask=None)

        # Regression output
        out = self.fc(attention_out)
        return out
