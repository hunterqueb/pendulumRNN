from load_gmat import *
import numpy as np  
from matplotlib import pyplot as plt

problemDim = 6
numRandSys = 10
mu = 398600  # Earth’s mu in km^3/s^2
R = 6371 # radius of earth in km
numMinProp = 60 * 24 # take a step 60 times in an hour and for 24 hours
numMinProp = 80 # take a step 60 times in an hour and for 24 hours
dt = 60.0 # step every 60 secs
elapsed = 0.0

# -----------configuration preliminaries----------------------------

mu = 398600

# Spacecraft
earthorb = gmat.Construct("Spacecraft", "EarthOrbiter") # create a spacecraft object named EarthOrbiter
earthorb.SetField("DateFormat", "UTCGregorian")
earthorb.SetField("Epoch", "20 Jul 2020 12:00:00.000") # set the epoch of the spacecraft

# Set the coordinate system and display state type
earthorb.SetField("CoordinateSystem", "EarthMJ2000Eq")
earthorb.SetField("DisplayStateType", "Keplerian")

# Spacecraft ballistic properties for the SRP and Drag models
earthorb.SetField("SRPArea", 2.5)
earthorb.SetField("Cr", 1.75)
earthorb.SetField("DragArea", 1.8)
earthorb.SetField("Cd", 2.1)
earthorb.SetField("DryMass", 80)

# Force model settings
fm = gmat.Construct("ForceModel", "FM")
fm.SetField("CentralBody", "Earth")

# A Full High-Fidelity 360x360 gravity field (incredibly slow)
earthgrav = gmat.Construct("GravityField")
earthgrav.SetField("BodyName","Earth")
earthgrav.SetField("PotentialFile", 'JGM2.cof')
earthgrav.SetField("Degree",8)
earthgrav.SetField("Order",8)


# Drag using Jacchia-Roberts
jrdrag = gmat.Construct("DragForce", "JRDrag")
jrdrag.SetField("AtmosphereModel","JacchiaRoberts")

# Build and set the atmosphere for the model
atmos = gmat.Construct("JacchiaRoberts", "Atmos")
jrdrag.SetReference(atmos)

# Construct Solar Radiation Pressure model
srp = gmat.Construct("SolarRadiationPressure", "SRP")

# Add forces into the ODEModel container
fm.AddForce(earthgrav)
fm.AddForce(jrdrag)
fm.AddForce(srp)

# Initialize propagator object
pdprop = gmat.Construct("Propagator","PDProp")

# Create and assign a numerical integrator for use in the propagation
gator = gmat.Construct("RungeKutta89", "Gator")
pdprop.SetReference(gator)

# Assign the force model contructed above
pdprop.SetReference(fm)

# Set some of the fields for the integration
pdprop.SetField("InitialStepSize", 60.0)
pdprop.SetField("Accuracy", 1.0e-12)
pdprop.SetField("MinStep", 0.0)

rng = np.random.default_rng()
rng.random()

## Generate Random Orbital Elements
earthorb.SetField("SMA", 7000) # km
earthorb.SetField("ECC", 0.05)
earthorb.SetField("INC", 10) # deg
earthorb.SetField("RAAN", 0) # deg
earthorb.SetField("AOP", 0) # deg
earthorb.SetField("TA", 0) # deg


statesArrayChemical = np.zeros((numRandSys,numMinProp,problemDim))

tank = gmat.Construct("ChemicalTank", "Fuel") # create a chemical tank with the name "Fuel"
thruster = gmat.Construct("ChemicalThruster", "Thruster") # create a chemical thruster with the name "Thruster"
thruster.SetField("DecrementMass", False)
thruster.SetField("Tank", "Fuel") # set the tank for the "Thruster" to use the "Fuel" object
earthorb.SetField("Tanks", "Fuel") # set possible tanks for the "Thruster" to use the "Fuel" object
earthorb.SetField("Thrusters", "Thruster") # set possible thrusters to use the "Thruster" object

gmat.Initialize()

# construct the burn force model
def setThrust(s, b):
    bf = gmat.FiniteThrust("Thrust")
    bf.SetRefObjectName(gmat.SPACECRAFT, s.GetName())
    bf.SetReference(b)
    gmat.ConfigManager.Instance().AddPhysicalModel(bf)
    return bf


burn = gmat.Construct("FiniteBurn", "TheBurn")
burn.SetField("Thrusters", "Thruster")
burn.SetSolarSystem(gmat.GetSolarSystem())
burn.SetSpacecraftToManeuver(earthorb)

burnForce = setThrust(earthorb, burn)


gmat.Initialize()


for i in range(1):
    earthorb.SetField("SMA", 7000) # km
    earthorb.SetField("ECC", 0.05)
    earthorb.SetField("INC", 10) # deg
    earthorb.SetField("RAAN", 0) # deg
    earthorb.SetField("AOP", 0) # deg
    earthorb.SetField("TA", 0) # deg

    tank.SetField("FuelMass", 200.0)

    thruster.SetField("C1",100000*rng.random()) # sets the first thrust coefficent. by default, chemical thrusters are set to a constant force output of 10 N and a 300 Ns impulse governed by a complex polynomial. See https://documentation.help/gmat/Thruster.html for specifics
    thruster.SetField("K1",3*rng.random())
    # Perform initializations
    gmat.Initialize()

    # Refresh the 'gator reference
    gator = pdprop.GetPropagator()

    gmat.Initialize()
    
    pdprop.AddPropObject(earthorb)
    pdprop.PrepareInternals()

    theThruster = earthorb.GetRefObject(gmat.THRUSTER, "Thruster")

    # -----------------------------
    # Finite Burn Specific Settings
    # -----------------------------
    # Turn on the thruster
    theThruster.SetField("IsFiring", True)
    earthorb.IsManeuvering(True)
    burn.SetSpacecraftToManeuver(earthorb)
    # # Add the thrust to the force model
    fm.AddForce(burnForce)
    psm = pdprop.GetPropStateManager()
    psm.SetProperty("MassFlow")
    # -----------------------------

    for j in range(numMinProp):
        gator.Step(dt)
        elapsed = elapsed + dt
        state = gator.GetState()
        statesArrayChemical[i,j,:] = state[0:6]
        gator.UpdateSpaceObject()

    fm = pdprop.GetODEModel()
    fm.DeleteForce(burnForce)
    theThruster.SetField("IsFiring", False)
    earthorb.IsManeuvering(False)
    pdprop.PrepareInternals()
    gator = pdprop.GetPropagator()

t = np.linspace(0,numMinProp*dt,len(statesArrayChemical[0,:,0]))

print(statesArrayChemical[0,:,0])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(statesArrayChemical[0,:,0],statesArrayChemical[0,:,1],statesArrayChemical[0,:,2],label='Chemical')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('3D Trajectory of Earth Orbiter')
ax.legend()
ax.axis('equal')

plt.show()
