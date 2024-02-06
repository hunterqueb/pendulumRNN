
import numpy as np
import matplotlib.pyplot as plt
import random
from quebUtils.integrators import myRK4Py
from matplotlib.animation import FuncAnimation
import nets
from torchinfo import summary

def linearOscialltor(t,y,p=None):

    dydt1 = y[1]
    dydt2 = -y[0]

    return np.array([dydt1,dydt2])

t0 = 0
tf = 10
numSteps = 100

t = np.linspace(t0,tf,numSteps)

# Example data: array of value pairs
data = np.zeros((5,5,numSteps,2))

for i in range(5):
    for j in range(5):
        theta = np.array([random.uniform(0.84, 0.86), random.uniform(-0.1, 0.1)])
        # data[0,i,j] = theta
        y = myRK4Py(linearOscialltor,theta,t,paramaters=None)
        data[i,j,:] = y

vmin = min(np.min(data), np.min(data))
vmax = max(np.max(data), np.max(data))

# Separate the pairs into two different arrays
# Initialize figure and axes
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Initialize two heatmaps
pos_ax = axes[0].imshow(data[:,:,0,0], interpolation='nearest', vmin=vmin, vmax=vmax)
vel_ax = axes[1].imshow(data[:,:,0,1], interpolation='nearest', vmin=vmin, vmax=vmax)

# Titles
axes[0].set_title("Heatmap of Position Values")
axes[1].set_title("Heatmap of Velocity Values")

time_text = fig.text(0.5, 0.04, '', ha='center')


# Function to update the content of the plots
def update(i):
    pos_ax.set_data(data[:,:,i,0])
    vel_ax.set_data(data[:,:,i,1])
    time_text.set_text(f"Time: {t[i]:.2f} seconds")

# Creating the animation
ani = FuncAnimation(fig, update, frames=numSteps, interval=50) # 50ms between frames

# Add colorbars
fig.colorbar(pos_ax, ax=axes[0])
fig.colorbar(vel_ax, ax=axes[1])

# Show the animation
plt.show()


#generate testing data
testData = np.zeros((2,2,numSteps,2))
for i in range(2):
    for j in range(2):
        theta = np.array([random.uniform(0.84, 0.86), random.uniform(-0.1, 0.1)])
        y = myRK4Py(linearOscialltor,theta,t,paramaters=None)
        testData[i,j,:] = y

model = nets.CNN_LSTM_SA()

summary(model)




# https://chat.openai.com/c/3a5bf74e-8f62-4435-b6e5-4c4042721439





