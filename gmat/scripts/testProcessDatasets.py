import numpy as np
import matplotlib.pyplot as plt

dt = 60

numMinProp = 30
numRandSys = 10000
orbitType = "leo"

dataLoc = "gmat/data/classification/" + orbitType + "/" + str(numMinProp) + "min-" + str(numRandSys)

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
ax.set_title('3D Trajectory of Earth Orbiter')
ax.legend(loc='lower left')
ax.axis('equal')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(statesArrayChemical[1,:,0],statesArrayChemical[1,:,1],statesArrayChemical[1,:,2],label='Chemical')
ax.plot(statesArrayElectric[1,:,0],statesArrayElectric[1,:,1],statesArrayElectric[1,:,2],label='Electric')
ax.plot(statesArrayImpBurn[1,:,0],statesArrayImpBurn[1,:,1],statesArrayImpBurn[1,:,2],label='Impulsive')
ax.plot(statesArrayNoThrust[1,:,0],statesArrayNoThrust[1,:,1],statesArrayNoThrust[1,:,2],label='No Thrust')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('3D Trajectory of Earth Orbiter')
ax.legend()
ax.axis('equal')


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


plotDiffFromNoThrust(statesArrayChemical, 'Chemical')
plotDiffFromNoThrust(statesArrayElectric, 'Electric')
plotDiffFromNoThrust(statesArrayImpBurn, 'Impulsive')
plt.show()