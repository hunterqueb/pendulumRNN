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

def transferLSTM(pretrainedModel,newModel):
    '''
    custom function to transfer knowledge of LSTM network from a pretrained model to a new model

    parameters: pretrainedModel - pretrained pytorch model with two LSTM layers
                newModel - untrained pytorch model with two LSTM layers
    '''
    newModel.lstm.load_state_dict(pretrainedModel.lstm.state_dict())
    newModel.self_attention.load_state_dict(pretrainedModel.self_attention.state_dict())

    # Freeze the weights of the LSTM layers
    for param in newModel.lstm.parameters():
        param.requires_grad = False
    for param in newModel.self_attention.parameters():
        param.requires_grad = False

    return newModel