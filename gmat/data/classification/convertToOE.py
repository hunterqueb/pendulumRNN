import numpy as np
import matplotlib.pyplot as plt

from qutils.orbital import ECI2OE
dt = 60

numMinProp = 100
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
del a

OEArrayChemical = np.zeros((numRandSys,numMinProp,7))
OEArrayElectric = np.zeros((numRandSys,numMinProp,7))
OEArrayImpBurn = np.zeros((numRandSys,numMinProp,7))
OEArrayNoThrust = np.zeros((numRandSys,numMinProp,7))

for i in range(numRandSys):
    for j in range(numMinProp):
        OEArrayChemical[i,j,:] = ECI2OE(statesArrayChemical[i,j,0:3],statesArrayChemical[i,j,3:6])
        OEArrayElectric[i,j,:] = ECI2OE(statesArrayElectric[i,j,0:3],statesArrayElectric[i,j,3:6])
        OEArrayImpBurn[i,j,:] = ECI2OE(statesArrayImpBurn[i,j,0:3],statesArrayImpBurn[i,j,3:6])
        OEArrayNoThrust[i,j,:] = ECI2OE(statesArrayNoThrust[i,j,0:3],statesArrayNoThrust[i,j,3:6])

np.savez(f"{dataLoc}/OEArrayChemical.npz", OEArrayChemical=OEArrayChemical)
np.savez(f"{dataLoc}/OEArrayElectric.npz", OEArrayElectric=OEArrayElectric)
np.savez(f"{dataLoc}/OEArrayImpBurn.npz", OEArrayImpBurn=OEArrayImpBurn)
np.savez(f"{dataLoc}/OEArrayNoThrust.npz", OEArrayNoThrust=OEArrayNoThrust)

plt.figure()
plt.plot(OEArrayChemical[0,:,0],label='Chemical')
plt.plot(OEArrayElectric[0,:,0],label='Electric')
plt.plot(OEArrayImpBurn[0,:,0],label='Impulsive')
plt.plot(OEArrayNoThrust[0,:,0],label='No Thrust')
plt.xlabel('Time (min)')
plt.ylabel('SMA (km)')
plt.grid()
plt.legend()
plt.title('Semi-Major Axis Time Series')
plt.show()