import numpy as np
import matplotlib.pyplot as plt
from qutils.orbital import dim2NonDim6,orbitalEnergy

dt = 60

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--numMinProp', type=int, default=10, help='Number of minutes of propagation')
parser.add_argument('--numRandSys', type=int, default=10000, help='Number of random systems')
parser.add_argument('--orbitType', type=str, default='vleo', help='Type of orbit')
parser.add_argument('--norm', action='store_true', help='Normalize the states')
args = parser.parse_args()
numMinProp = args.numMinProp
numRandSys = args.numRandSys
orbitType = args.orbitType
norm = args.norm

print(f"Processing datasets for {orbitType} with {numMinProp} minutes and {numRandSys} random systems.")

import yaml
with open("data.yaml", 'r') as f:
    dataConfig = yaml.safe_load(f)
dataLoc = dataConfig["classification"] + orbitType + "/" + str(numMinProp) + "min-" + str(numRandSys)

# get npz files in folder and load them into script

a = np.load(f"{dataLoc}/statesArrayChemical.npz")
statesArrayChemical = a['statesArrayChemical']

a = np.load(f"{dataLoc}/statesArrayElectric.npz")
statesArrayElectric = a['statesArrayElectric']

a = np.load(f"{dataLoc}/statesArrayImpBurn.npz")
statesArrayImpBurn = a['statesArrayImpBurn']

a = np.load(f"{dataLoc}/statesArrayNoThrust.npz")
statesArrayNoThrust = a['statesArrayNoThrust']

print(statesArrayChemical.shape)
print(statesArrayElectric.shape)
print(statesArrayImpBurn.shape)
print(statesArrayNoThrust.shape)

if norm:
    for i in range(statesArrayChemical.shape[0]):
        statesArrayChemical[i,:,:] = dim2NonDim6(statesArrayChemical[i,:,:])
        statesArrayElectric[i,:,:] = dim2NonDim6(statesArrayElectric[i,:,:])
        statesArrayImpBurn[i,:,:] = dim2NonDim6(statesArrayImpBurn[i,:,:])
        statesArrayNoThrust[i,:,:] = dim2NonDim6(statesArrayNoThrust[i,:,:])

energyChemical = np.zeros((statesArrayChemical.shape[0],statesArrayChemical.shape[1],1))
energyElectric= np.zeros((statesArrayChemical.shape[0],statesArrayChemical.shape[1],1))
energyImpBurn= np.zeros((statesArrayChemical.shape[0],statesArrayChemical.shape[1],1))
energyNoThrust= np.zeros((statesArrayChemical.shape[0],statesArrayChemical.shape[1],1))
for i in range(statesArrayChemical.shape[0]):
    energyChemical[i,:,0] = orbitalEnergy(statesArrayChemical[i,:,:])
    energyElectric[i,:,0] = orbitalEnergy(statesArrayElectric[i,:,:])
    energyImpBurn[i,:,0] = orbitalEnergy(statesArrayImpBurn[i,:,:])
    energyNoThrust[i,:,0] = orbitalEnergy(statesArrayNoThrust[i,:,:])

t = np.linspace(0,100*dt,len(statesArrayChemical[0,:,0]))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(statesArrayChemical[0,:,0],statesArrayChemical[0,:,1],statesArrayChemical[0,:,2],label='Chemical')
ax.plot(statesArrayElectric[0,:,0],statesArrayElectric[0,:,1],statesArrayElectric[0,:,2],label='Electric')
ax.plot(statesArrayImpBurn[0,:,0],statesArrayImpBurn[0,:,1],statesArrayImpBurn[0,:,2],label='Impulsive')
ax.plot(statesArrayNoThrust[0,:,0],statesArrayNoThrust[0,:,1],statesArrayNoThrust[0,:,2],label='No Thrust')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('3D Trajectory of a Single Earth Orbiter')
ax.legend(loc='lower left')
ax.axis('equal')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i in range(20):
    ax.plot(statesArrayChemical[i,:,0],statesArrayChemical[i,:,1],statesArrayChemical[i,:,2],label='Chemical',color='C0')
    ax.plot(statesArrayElectric[i,:,0],statesArrayElectric[i,:,1],statesArrayElectric[i,:,2],label='Electric',color='C1')
    ax.plot(statesArrayImpBurn[i,:,0],statesArrayImpBurn[i,:,1],statesArrayImpBurn[i,:,2],label='Impulsive',color='C2')
    ax.plot(statesArrayNoThrust[i,:,0],statesArrayNoThrust[i,:,1],statesArrayNoThrust[i,:,2],label='No Thrust',color='C3')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('3D Trajectory of 20 Earth Orbiters')
from matplotlib.lines import Line2D
colors = ['C0', 'C1', 'C2', 'C3']
lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in colors]
labels = ['Chemical Thrust', 'Electrical Thrust', 'Impulsive Thrust', 'No Thrust']
ax.legend(lines, labels)
ax.axis('equal')

plt.figure()
for i in range(20):
    plt.plot(t, energyChemical[i,:,0], label='Chemical',color='C0')
    plt.plot(t, energyElectric[i,:,0], label='Electric',color='C1')
    plt.plot(t, energyImpBurn[i,:,0], label='Impulsive',color='C2')
    plt.plot(t, energyNoThrust[i,:,0], label='No Thrust',color='C3')
plt.legend(lines, labels)
plt.grid()
plt.xlabel('Time (s)')
plt.title("Energy of 20 Earth Orbiters")

plt.figure()
plt.plot(t, statesArrayChemical[0,:,0], label='Chemical X')
plt.plot(t, statesArrayElectric[0,:,0], label='Electric X')
plt.plot(t, statesArrayImpBurn[0,:,0], label='Impulsive X')
plt.plot(t, statesArrayNoThrust[0,:,0], label='No Thrust X')
plt.plot(t, statesArrayChemical[0,:,1], label='Chemical Y')
plt.plot(t, statesArrayElectric[0,:,1], label='Electric Y')
plt.plot(t, statesArrayImpBurn[0,:,1], label='Impulsive Y')
plt.plot(t, statesArrayNoThrust[0,:,1], label='No Thrust Y')
plt.plot(t, statesArrayChemical[0,:,2], label='Chemical Z')
plt.plot(t, statesArrayElectric[0,:,2], label='Electric Z')
plt.plot(t, statesArrayImpBurn[0,:,2], label='Impulsive Z')
plt.plot(t, statesArrayNoThrust[0,:,2], label='No Thrust Z')
plt.xlabel('Time (s)')
plt.ylabel('Position (km)')
plt.title('Position vs Time for Different Thruster Profiles')
plt.legend(loc='lower left')
plt.grid()

def plotDiffFromNoThrust(statesArray, label):
    plt.figure()
    plt.plot(t, statesArray[0,:,0]-statesArrayNoThrust[0,:,0], label=label+' X')
    plt.plot(t, statesArray[0,:,1]-statesArrayNoThrust[0,:,1], label=label+' Y')
    plt.plot(t, statesArray[0,:,2]-statesArrayNoThrust[0,:,2], label=label+' Z')
    plt.xlabel('Time (s)')
    plt.ylabel('Position Difference from No Thrust (km)')
    plt.title(f'Position Difference from No Thrust vs Time for {label} Thruster Profile')
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(t, statesArray[0,:,3]-statesArrayNoThrust[0,:,3], label=label+' VX')
    plt.plot(t, statesArray[0,:,4]-statesArrayNoThrust[0,:,4], label=label+' VY')
    plt.plot(t, statesArray[0,:,5]-statesArrayNoThrust[0,:,5], label=label+' VZ')
    plt.xlabel('Time (s)')
    plt.ylabel('Position Difference from No Thrust (km/s)')
    plt.title(f'Position Difference from No Thrust vs Time for {label} Thruster Profile')
    plt.legend()
    plt.grid()


# plotDiffFromNoThrust(statesArrayChemical, 'Chemical')
# plotDiffFromNoThrust(statesArrayElectric, 'Electric')
# plotDiffFromNoThrust(statesArrayImpBurn, 'Impulsive')
plt.show()