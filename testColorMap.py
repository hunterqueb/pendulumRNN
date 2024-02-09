import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from quebUtils.integrators import myRK4Py
from matplotlib.animation import FuncAnimation
import nets
from torchinfo import summary
import torchvision
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

def linearOscialltor(t,y,p=None):

    dydt1 = y[1]
    dydt2 = -y[0]

    return np.array([dydt1,dydt2])

t0 = 0
tf = 10
numSteps = 100

t = np.linspace(t0,tf,numSteps)

# Example data: array of value pairs
num_instances = 100


data = np.zeros((num_instances,5,5,numSteps,2))
for k in range(num_instances):
    for i in range(5):
        for j in range(5):
            theta = np.array([random.uniform(0.84, 0.86), random.uniform(-0.1, 0.1)])
            # data[0,i,j] = theta
            y = myRK4Py(linearOscialltor,theta,t,paramaters=None)
            data[k,i,j,:] = y

vmin = min(np.min(data), np.min(data))
vmax = max(np.max(data), np.max(data))

# Separate the pairs into two different arrays
# Initialize figure and axes
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Initialize two heatmaps
pos_ax = axes[0].imshow(data[0,:,:,0,0], interpolation='nearest', vmin=vmin, vmax=vmax)
vel_ax = axes[1].imshow(data[0,:,:,0,1], interpolation='nearest', vmin=vmin, vmax=vmax)

# Titles
axes[0].set_title("Heatmap of Position Values")
axes[1].set_title("Heatmap of Velocity Values")

time_text = fig.text(0.5, 0.04, '', ha='center')


# Function to update the content of the plots
def update(i):
    pos_ax.set_data(data[0,:,:,i,0])
    vel_ax.set_data(data[0,:,:,i,1])
    time_text.set_text(f"Time: {t[i]:.2f} seconds")

# Creating the animation
ani = FuncAnimation(fig, update, frames=numSteps, interval=50) # 50ms between frames

# Add colorbars
fig.colorbar(pos_ax, ax=axes[0])
fig.colorbar(vel_ax, ax=axes[1])

# Show the animation
plt.show()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)




model = nets.CNN_LSTM_SA().double().to(device)

summary(model)


data = np.random.rand(num_instances, 5, 5, numSteps, 2) # Random data for demonstration
targets = np.random.rand(num_instances) # Random target values

# Reshape data to [num_instances, 100, 2, 5, 5]
data = data.transpose(0, 1, 4, 2, 3)

# Shuffle the dataset
indices = np.arange(num_instances)
np.random.shuffle(indices)
data = data[indices]
targets = targets[indices]

# Split the data into training and testing sets
test_size = 0.2
split_index = int(num_instances * (1 - test_size))
x_train, x_test = data[:split_index], data[split_index:]
y_train, y_test = targets[:split_index], targets[split_index:]

# Normalize or standardize the data
# Here is a simple normalization example
mean = x_train.mean(axis=(0, 1, 2, 3), keepdims=True)
std = x_train.std(axis=(0, 1, 2, 3), keepdims=True)
x_train = (x_train - mean) / std
X_test = (x_test - mean) / std

x_train = torch.tensor(x_train).double().to(device)
X_test = torch.tensor(X_test).double().to(device)

model(x_train)
model(x_train)
model(x_train)


print()
# https://chat.openai.com/c/3a5bf74e-8f62-4435-b6e5-4c4042721439
# https://machinelearningmastery.com/building-a-convolutional-neural-network-in-pytorch/




