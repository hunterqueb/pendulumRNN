import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data

from qutils.integrators import ode45
from qutils.ml import printModelParmSize, getDevice, create_datasets, genPlotPrediction, trainModel

from qutils.mamba import Mamba, MambaConfig

from qutils.mlExtras import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight

activationArea = 'input'
layer_path = "layers"
DEBUG = True

mambaLayerAttributes = ["in_proj","conv1d","x_proj","dt_proj","out_proj"]

def custom_unravel_index(indices: torch.LongTensor, shape: tuple):
    """
    Mimics torch.unravel_index(indices, shape) without directly calling it.
    
    Args:
        indices (torch.LongTensor): A 1D tensor of flat indices.
        shape (tuple): The target shape to unravel into.
    
    Returns:
        torch.LongTensor: A 2D tensor of shape (len(shape), indices.size(0)),
                          containing the unraveled coordinates.
    """
    # Flatten the N-D indices to a 1D tensor
    flat_indices = indices.view(-1).long()
    
    # Perform the unraveling on the flattened tensor
    coords = []
    for dim in reversed(shape):
        coords.append(flat_indices % dim)
        flat_indices = flat_indices // dim
    
    # Reverse coordinates since we iterated from last dimension to first
    coords.reverse()
    
    # Now reshape each coordinate vector back to the original indices shape
    # so that coords[i] has the same shape as indices (except it gives the
    # i-th dimension's coordinate)
    coords = [c.reshape(indices.shape) for c in coords]
    
    # Finally, stack along dim=0 to get a shape of (len(shape), *indices.shape)
    return torch.stack(coords, dim=0)


def findSuperActivation(model,test_in,input_or_output=activationArea,layer_path=layer_path):
    module_name = "mixer"
    all_activations = {}
    all_hooks = []

    def get_activations(layer_index):
        def hook(model, inputs, outputs):
            hidden_states = inputs if input_or_output == "input" else outputs
            all_activations.setdefault(layer_index, {})[f"{module_name}_{input_or_output}_hidden_states"] = hidden_states
        return hook   

    def get_layers(model, layer_path):
        attributes = layer_path.split('.')
        layers = model
        for attr in attributes:
            layers = getattr(layers, attr)
        return layers


    attributes = module_name.split('.') if module_name != "layer" else []
    layers = get_layers(model, layer_path)

    for layer_index, layer in enumerate(layers):
        mixerAttr = layer
        valid = True
        for attr in attributes:
            if hasattr(mixerAttr, attr):
                mixerAttr = getattr(mixerAttr, attr)
                layer_index = 0
                for innerAttr in mambaLayerAttributes:
                    current_attr = getattr(mixerAttr, innerAttr)
                    hook = current_attr.register_forward_hook(get_activations(layer_index))
                    all_hooks.append(hook)
                    layer_index += 1
            else:
                valid = False
                break
        


    model.eval()
    with torch.no_grad():
        model(test_in)
    for hook in all_hooks:
        hook.remove()
    top1_values_all_layers = []
    top1_indexes_all_layers = []
    for layer_index, outputs in all_activations.items():
        values = outputs[f'{module_name}_{input_or_output}_hidden_states']
        tensor = values[0] if isinstance(values, tuple) else values
        tensor = tensor.detach().cpu()
        tensor_abs = tensor.abs().float()

        # tensor2d = tensor.abs().reshape((tensor.shape[0],tensor.shape[2]))


        max_value, max_index = torch.max(tensor_abs, 0)
        max_index = custom_unravel_index(max_index, tensor.shape)
        top1_values_all_layers.append(tensor[max_index])
        top1_indexes_all_layers.append(max_index)


    return top1_values_all_layers, top1_indexes_all_layers



plotOn = True
printoutSuperweight = True

problemDim = 4 

device = getDevice()

m1 = 1
m2 = m1
l1 = 1
l2 = l1
g = 9.81
parameters = np.array([m1,m2,l1,l2,g])

def doublePendulumODE(t,y,p=parameters):
    # p = [m1,m2,l1,l2,g]
    m1 = p[0]
    m2 = p[1]
    l1 = p[2]
    l2 = p[3]
    g = p[4]

    theta1 = y[0]
    theta2 = y[2]

    dydt1 = y[1] #theta1dot

    dydt2 = (m2*g*np.sin(theta2)*np.cos(theta1-theta2) - m2*np.sin(theta1-theta2)*(l1*y[1]**2*np.cos(theta1-theta2) + l2*y[3]**2)
            - (m1+m2)*g*np.sin(theta1)) / l1 / (m1 + m2*np.sin(theta1-theta2)**2) #theta1ddot

    dydt3 = y[3] #theta2dot

    dydt4 = ((m1+m2)*(l1*y[1]**2*np.sin(theta1-theta2) - g*np.sin(theta2) + g*np.sin(theta1)*np.cos(theta1-theta2))
            + m2*l2*y[3]**2*np.sin(theta1-theta2)*np.cos(theta1-theta2)) / l2 / (m1 + m2*np.sin(theta1-theta2)**2) #theta2ddot

    return np.array((dydt1,dydt2,dydt3,dydt4))

theta1_0 = np.radians(80)
theta2_0 = np.radians(135)
thetadot1_0 = np.radians(-1)
thetadot2_0 = np.radians(0.7)
initialConditions = np.array([theta1_0,thetadot1_0,theta2_0,thetadot2_0],dtype=np.float64)

# initialConditions = np.radians(np.random.uniform(-180, 180, (problemDim,)))

tStart = 0
tEnd = 20
tSpan = np.array([tStart,tEnd])
dt = 0.01
tSpanExplicit = np.linspace(tStart,tEnd,int(tEnd / dt))

t , numericResult = ode45(doublePendulumODE,[tStart,tEnd],initialConditions,tSpanExplicit)

output_seq = numericResult

# hyperparameters
n_epochs = 5
# lr = 5*(10**-5)
# lr = 0.85
lr = 0.8
lr = 0.08
lr = 0.004
lr = 0.0001
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = 1
p_motion_knowledge = 0.5


train_size = int(len(output_seq) * p_motion_knowledge)
# train_size = 2
test_size = len(output_seq) - train_size

train_in,train_out,test_in,test_out = create_datasets(output_seq,1,train_size,device)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

# initilizing the model, criterion, and optimizer for the data
config = MambaConfig(d_model=problemDim, n_layers=num_layers)
model = Mamba(config).to(device).double()
# model = LSTM(input_size,10,output_size,num_layers,0).double().to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = F.smooth_l1_loss

trainModel(model,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutToc=False)


from qutils.plot import plotStatePredictions

networkPrediction = plotStatePredictions(model,t,output_seq,train_in,test_in,train_size,test_size,1,states=('\\theta_1','\\theta_2','\\theta_1_dot','\\theta_2_dot'),units=('rad','rad','rad/s','rad/s'),plotOn=not DEBUG)

# superweight training function here
magnitude, index = findSuperActivation(model,test_in)

# spikes_input = [i for i, value in enumerate(magnitude) if abs(value.norm()) > 50]
# print(f"Activation spikes")
# for i in spikes_input:
#     spike_index = index[i]
#     print(f" - layer {i}, value {magnitude[i]}, index {tuple(i.item() for i in spike_index)}")


# plotDecAccs(decAcc,t,problemDim)
errorAvg = np.nanmean(abs(networkPrediction-output_seq) * 90 / np.pi, axis=0)
print("Average error of each dimension:")
unitLabels = ['deg','deg/s','deg','deg/s']
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg} {unitLabels[i-1]}")

printModelParmSize(model)

if printoutSuperweight is True:
    printoutMaxLayerWeight(model)
    getSuperWeight(model)
    plotSuperWeight(model)

if plotOn is True:
    # Plot input activations
    plt.figure(figsize=(5,3.5))
    for i in range(len(magnitude)):
        plt.plot(i, magnitude[i].norm(), color='blue', marker='o', markersize=5)
    plt.xlabel('Layer')
    plt.xticks((0,1,2,3,4),mambaLayerAttributes)
    plt.ylabel('Max Activation Value')
    plt.grid()
    plt.title(f"{activationArea} Activation")

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(output_seq[:,0],output_seq[:,1],'r',label = "Truth")
    plt.plot(networkPrediction[:,0],networkPrediction[:,1],'b',label = "NN")
    plt.xlabel('Theta 1')
    plt.ylabel('Theta 1 Dot')
    plt.axis('equal')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(output_seq[:,2],output_seq[:,3],'r',label = "Truth")
    plt.plot(networkPrediction[:,2],networkPrediction[:,3],'b',label = "NN")
    plt.xlabel('Theta 2')
    plt.ylabel('Theta 2 Dot')
    plt.axis('equal')
    plt.legend()
    plt.grid()

    plt.show()


