import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim.optimizer import Optimizer
import math
import torch.distributed as dist

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Adam_mini(Optimizer):
    def __init__(
        self,
        model=None,
        weight_decay=0.1,
        lr=1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        zero_3=False,
        n_embd = None,
        n_head = None,
        n_query_groups = None
    ):
        '''
        model: the model you are training.

        zero_3: set to True if you are using zero_3 in Deepspeed, or if you are using model parallelism with more than 1 GPU. Set to False if otherwise.
        
        n_embd: number of embedding dimensions. Could be unspecified if you are training non-transformer models.
        
        n_head: number of attention heads. Could be unspecified if you are training non-transformer models.
        
        n_query_groups: number of query groups in Group query Attention. If not specified, it will be equal to n_head. Could be unspecified if you are training non-transformer models.
        '''
       

        self.n_embd = n_embd
        self.n_head = n_head
        if n_query_groups is not None:
            self.n_query_groups = n_query_groups
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

        self.model = model
        self.world_size = torch.cuda.device_count()
        self.zero_optimization_stage = 0
        if zero_3:
            self.zero_optimization_stage = 3
            print("Adam-mini is using zero_3")
        optim_groups = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                dic = {}
                dic["name"] = name
                dic["params"] = param
                if ("norm" in name or "ln_f" in name):
                    dic["weight_decay"] = 0
                else:
                    dic["weight_decay"] = weight_decay
                
                if ("self_attn.k_proj.weight" in name or "self_attn.q_proj.weight" in name):
                    dic["parameter_per_head"] = self.n_embd * self.n_embd // self.n_head
                
                if ("attn.attn.weight" in name or "attn.qkv.weight" in name):
                    dic["n_head"] = self.n_head
                    dic["q_per_kv"] = self.n_head // self.n_query_groups

                optim_groups.append(dic)

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)

        super(Adam_mini, self).__init__(optim_groups, defaults)


    def step(self):
        with torch.no_grad():
            for group in self.param_groups:
                beta1 = group["beta1"]
                beta2 = group["beta2"]
                lr = group["lr"]
                name = group["name"]
                epsilon = group["epsilon"]
                
                for p in group["params"]:
                    state = self.state[p]
                    if ("embed_tokens" in name or "wte" in name or "lm_head" in name):
                        if p.grad is None:
                            continue
                        if len(state) == 0:
                            state["m"] = torch.zeros_like(p.data).to(torch.float32)
                            state["iteration"] = 0
                            state["v"] = torch.zeros_like(p.data).to(torch.float32)

                        grad = p.grad.data.to(torch.float32)
                        state["v"].mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                        state["iteration"] += 1
                        if group["weight_decay"] != 0:
                            p.data.mul_(1 - lr * group["weight_decay"])

                        state["m"].lerp_(grad, 1 - beta1)

                        bias_correction_1 = 1 - beta1 ** state["iteration"]
                        bias_correction_2 = 1 - beta2 ** state["iteration"]
                        bias_correction_2_sqrt = math.sqrt(bias_correction_2)

                        h = (state["v"].sqrt() / bias_correction_2_sqrt).add_(epsilon)
                        stepsize = lr/ bias_correction_1
                        p.addcdiv_(state["m"], h, value=-stepsize)

                    elif ("self_attn.k_proj.weight" in name or "self_attn.q_proj.weight" in name or "attn.wq.weight" in name or "attn.wk.weight" in name):
                        if p.grad is None:
                            continue
                        dim = group["parameter_per_head"]
                        if (len(state)==0):
                            state["m"]  =  torch.zeros_like(p.data).to(torch.float32)
                            state["m"] = state["m"].view(-1, dim)
                            state['head'] = state['m'].shape[0]
                            state["iteration"] = 0
                            state["vmean"] = torch.zeros(state['head']).to(device)

                        grad = p.grad.data.to(torch.float32)
                        head = state['head']
                        grad = grad.view(head, dim)

                        tmp_lr = torch.mean(grad*grad, dim = 1).to(device)
                        state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                        v = state["vmean"]

                        state["iteration"] += 1
                        if group["weight_decay"] != 0:
                            p.data.mul_(1 - lr * group["weight_decay"])

                        state["m"].lerp_(grad, 1 - beta1)

                        bias_correction_1 = 1 - beta1 ** state["iteration"]
                        bias_correction_2 = 1 - beta2 ** state["iteration"]
                        bias_correction_2_sqrt = math.sqrt(bias_correction_2)

                        h = (v.sqrt() / bias_correction_2_sqrt).add_(epsilon)    
                        stepsize = ((1/bias_correction_1) / h).view(head,1)

                        update = state["m"] * (stepsize.to(state['m'].device))

                        if p.dim() > 1:
                            d0, d1 = p.size()
                            update = update.view(d0, d1)
                        else: 
                            update = update.view(-1)

                        update.mul_(lr)
                        p.add_(-update)
                        
                    elif ("attn.attn.weight" in name or "attn.qkv.weight" in name): 
                        if p.grad is None:
                            continue
                        if (len(state)==0):
                            state["m"]  =  torch.zeros_like(p.data).to(torch.float32)
                            state["m"] = state["m"].view(group["n_head"], group["q_per_kv"] + 2, -1)
                            state["iteration"] = 0
                            state["vmean"] = torch.zeros(group["n_head"], group["q_per_kv"]+2).to(device)
                            

                        grad = p.grad.data.to(torch.float32)
                        grad = grad.view(group["n_head"], group["q_per_kv"] + 2, -1) 

                        tmp_lr = torch.mean(grad*grad, dim = 2).to(device)
                        state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                        v = state["vmean"]
                
                        state["iteration"] += 1
                        if group["weight_decay"] != 0:
                            p.data.mul_(1 - lr * group["weight_decay"])

                        state["m"].lerp_(grad, 1 - beta1)


                        bias_correction_1 = 1 - beta1 ** state["iteration"]
                        bias_correction_2 = 1 - beta2 ** state["iteration"]
                        bias_correction_2_sqrt = math.sqrt(bias_correction_2)

                        h = (v.sqrt() / bias_correction_2_sqrt).add_(epsilon)    
                        stepsize = ((1/bias_correction_1) / h).view(group["n_head"],group["q_per_kv"]+2,1)
                                                    
   
                        update = state["m"] * (stepsize.to(state['m'].device))
            
                        if p.dim() > 1:
                            d0, d1 = p.size()
                            update = update.view(d0, d1)
                        else: 
                            update = update.view(-1)
        
                        update.mul_(lr)
                        p.add_(-update)

                        
                    else:        
                        if (len(state)==0):                   
                            dimension = torch.tensor(p.data.numel()).to(device).to(torch.float32)
                            reduced = False
                            if (self.world_size > 1) and (self.zero_optimization_stage == 3):
                                tensor_list = [torch.zeros_like(dimension) for _ in range(self.world_size)]
                                dist.all_gather(tensor_list, dimension)
                                s = 0
                                dimension = 0
                                for d in tensor_list:
                                    if (d>0):
                                        s = s + 1
                                    dimension = dimension + d
                                if (s>=2):
                                    reduced = True
                            
                            state["m"] = torch.zeros_like(p.data).to(torch.float32)
                            state["iteration"] = 0
                            state["reduced"] = reduced
                            state["vmean"] = torch.tensor(0.0).to(device)                                
                            state["dimension"] = dimension.item()
                        if p.grad is None:
                            tmp_lr = torch.tensor(0.0).to(device)
                        else:
                            grad = p.grad.data.to(torch.float32)
                            tmp_lr = torch.sum(grad*grad)                               
                        if (state["reduced"]):
                            dist.all_reduce(tmp_lr, op=dist.ReduceOp.SUM)
                    
                        tmp_lr = tmp_lr / (state["dimension"])
                        tmp_lr = tmp_lr.to(grad.device)
                        if (p.grad is None):
                            continue
                        if group["weight_decay"] != 0:
                            p.data.mul_(1 - lr * group["weight_decay"])
                        state["iteration"] += 1
                        state["m"].lerp_(grad, 1 - beta1)

                        bias_correction_1 = 1 - beta1 ** state["iteration"]
                        bias_correction_2 = 1 - beta2 ** state["iteration"]
                        bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                        state["vmean"] = (1 - beta2) * tmp_lr + beta2 * state["vmean"]
                        h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(epsilon)    

                        stepsize = (1 / bias_correction_1) / h
                        update = state["m"] * (stepsize.to(state['m'].device))
                        update.mul_(lr)
                        p.add_(-update)    
                    
     

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



def transferMamba(pretrainedModel,newModel,trainableLayers = [True,False,False]):
    '''
    custom function to transfer knowledge of a mamba network from a pretrained model to a new model
    the mamba network is from https://github.com/alxndrTL/mamba.py

    parameters: pretrainedModel - pretrained pytorch mamba model with one state space layer
                newModel - untrained pytorch mamba model with one state space layer
    '''
    # deltaBC is calced simultaneously here!
    # model.layers[0].mixer.x_proj.state_dict()

    # load the parameters from the old model to the new, and set all parameters to untrainable
    newModel.load_state_dict(pretrainedModel.state_dict())
    for param in newModel.parameters():
        param.requires_grad = False

    for param in newModel.layers[0].mixer.conv1d.parameters():
        param.requires_grad = True

    # trainanle A matrix
    newModel.layers[0].mixer.A_log.requires_grad = False

    # trainable deltaBC matrix
    for param in newModel.layers[0].mixer.x_proj.parameters():
        param.requires_grad = trainableLayers[0]

    for param in newModel.layers[0].mixer.dt_proj.parameters():
        param.requires_grad = trainableLayers[1]

    # probably not the best to transfer, this is the projection from the latent state space back to the output space
    for param in newModel.layers[0].mixer.out_proj.parameters():
        param.requires_grad = trainableLayers[2]


    return newModel

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


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_value, heads=1):
        super(TransformerModel, self).__init__()

        # Transformer layer
        self.transformer = nn.Transformer(d_model=input_dim, nhead=heads, num_encoder_layers=num_layers, dropout=dropout_value)

        # Fully connected layer
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Pass data through Transformer layer
        transformer_out = self.transformer(x)

        # Pass data through fully connected layer
        final_out = self.fc(transformer_out)

        return final_out

