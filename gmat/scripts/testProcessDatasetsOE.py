import numpy as np
import matplotlib.pyplot as plt

dt = 60

numMinProp = 100
numRandSys = 10000
orbitType = "leo"

dataLoc = "gmat/data/classification/" + orbitType + "/" + str(numMinProp) + "min-" + str(numRandSys)

# get npz files in folder and load them into script

a = np.load(f"{dataLoc}/OEArrayChemical.npz")
statesArrayChemical = a['OEArrayChemical']

a = np.load(f"{dataLoc}/OEArrayElectric.npz")
statesArrayElectric = a['OEArrayElectric']

a = np.load(f"{dataLoc}/OEArrayImpBurn.npz")
statesArrayImpBurn = a['OEArrayImpBurn']

a = np.load(f"{dataLoc}/OEArrayNoThrust.npz")
statesArrayNoThrust = a['OEArrayNoThrust']

print(statesArrayChemical.shape)
print(statesArrayElectric.shape)
print(statesArrayImpBurn.shape)
print(statesArrayNoThrust.shape)

t = np.linspace(0,numMinProp*dt,len(statesArrayChemical[0,:,0]))

plt.figure()
plt.plot(t, statesArrayChemical[0,:,0], label='Chemical')
plt.plot(t, statesArrayElectric[0,:,0], label='Electric')
plt.plot(t, statesArrayImpBurn[0,:,0], label='Impulsive')
plt.plot(t, statesArrayNoThrust[0,:,0], label='No Thrust')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.title("Semi-Major Axis")


plt.figure()

plt.plot(t, statesArrayChemical[0,:,1], label='Chemical')
plt.plot(t, statesArrayElectric[0,:,1], label='Electric')
plt.plot(t, statesArrayImpBurn[0,:,1], label='Impulsive')
plt.plot(t, statesArrayNoThrust[0,:,1], label='No Thrust')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.title("Eccentricity")

plt.figure()

plt.plot(t, statesArrayChemical[0,:,2], label='Chemical')
plt.plot(t, statesArrayElectric[0,:,2], label='Electric')
plt.plot(t, statesArrayImpBurn[0,:,2], label='Impulsive')
plt.plot(t, statesArrayNoThrust[0,:,2], label='No Thrust')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.title("Inclination")

plt.figure()
plt.plot(t, statesArrayChemical[0,:,3], label='Chemical')
plt.plot(t, statesArrayElectric[0,:,3], label='Electric')
plt.plot(t, statesArrayImpBurn[0,:,3], label='Impulsive')
plt.plot(t, statesArrayNoThrust[0,:,3], label='No Thrust')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.title("RAAN")

plt.figure()

plt.plot(t, statesArrayChemical[0,:,4], label='Chemical')
plt.plot(t, statesArrayElectric[0,:,4], label='Electric')
plt.plot(t, statesArrayImpBurn[0,:,4], label='Impulsive')
plt.plot(t, statesArrayNoThrust[0,:,4], label='No Thrust')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.title("Arg of Perigee")

plt.figure()

plt.plot(t, statesArrayChemical[0,:,5], label='Chemical')
plt.plot(t, statesArrayElectric[0,:,5], label='Electric')
plt.plot(t, statesArrayImpBurn[0,:,5], label='Impulsive')
plt.plot(t, statesArrayNoThrust[0,:,5], label='No Thrust')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.title("Mean Anomaly")

plt.figure()

plt.plot(t, statesArrayChemical[0,:,6], label='Chemical')
plt.plot(t, statesArrayElectric[0,:,6], label='Electric')
plt.plot(t, statesArrayImpBurn[0,:,6], label='Impulsive')
plt.plot(t, statesArrayNoThrust[0,:,6], label='No Thrust')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.title("Period")

# def plotDiffFromNoThrust(statesArray, label):
#     plt.figure()
#     plt.plot(t, statesArray[0,:,0]-statesArrayNoThrust[0,:,0], label=label+' X')
#     plt.plot(t, statesArray[0,:,1]-statesArrayNoThrust[0,:,1], label=label+' Y')
#     plt.plot(t, statesArray[0,:,2]-statesArrayNoThrust[0,:,2], label=label+' Z')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Position Difference from No Thrust (km)')
#     plt.title(f'Position Difference from No Thrust vs Time for {label} Thruster Profile')
#     plt.legend()
#     plt.grid()

#     plt.figure()
#     plt.plot(t, statesArray[0,:,3]-statesArrayNoThrust[0,:,3], label=label+' VX')
#     plt.plot(t, statesArray[0,:,4]-statesArrayNoThrust[0,:,4], label=label+' VY')
#     plt.plot(t, statesArray[0,:,5]-statesArrayNoThrust[0,:,5], label=label+' VZ')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Position Difference from No Thrust (km/s)')
#     plt.title(f'Position Difference from No Thrust vs Time for {label} Thruster Profile')
#     plt.legend()
#     plt.grid()


# plotDiffFromNoThrust(statesArrayChemical, 'Chemical')
# plotDiffFromNoThrust(statesArrayElectric, 'Electric')
# plotDiffFromNoThrust(statesArrayImpBurn, 'Impulsive')
plt.show()