from load_gmat import *
import numpy as np  
from matplotlib import pyplot as plt
import argparse

problemDim = 6
mu = 398600  # Earth’s mu in km^3/s^2
R = 6378 # radius of earth in km
dt = 60.0 # step every 60 secs
elapsed = 0.0


parser = argparse.ArgumentParser(description='GMAT Dataset Generation for Different Thruster Types')
parser.add_argument('--deltaV', type=float, default=1.0, help='Delta V for impulsive burn in km/s')
parser.add_argument('--numRandSys', type=int, default=1000, help='Number of random systems to generate')
parser.add_argument('--numMinProp', type=int, default=100, help='Number of minutes to propagate')
parser.add_argument('--chemThrust',type=float, default=10, help='Chemical thrust coefficent')
parser.add_argument('--elecThrust',type=float, default=-5.19082, help='Electric thrust coefficent')
parser.add_argument('--no-plot', dest="plotOn", action='store_false', help='Plot the results')
parser.set_defaults(plotOn=True)
parser.add_argument('--log', dest='echoLogFile', action='store_true', help='Log the GMAT output to console')
parser.set_defaults(echoLogFile=False)
parser.add_argument('--no-save', dest='saveOn', action='store_false', help='Save the results to file')
parser.set_defaults(saveOn=True)
parser.add_argument('--slurm', dest='slurm', action='store_true', help='Run in slurm mode')
parser.set_defaults(slurm=False)
parser.add_argument('--propType', type=str, default='chem', help='Type of propagation to perform: chem, elec, imp, none')
parser.add_argument("--lowerAlt", type=float, default=200, help="Lower altitude bound in km")
parser.add_argument("--upperAlt", type=float, default=250, help="Upper altitude bound in km")

args = parser.parse_args()

deltaV = args.deltaV
numRandSys = args.numRandSys
numMinProp = args.numMinProp
chemThrust = args.chemThrust
elecThrust = args.elecThrust
plotOn = args.plotOn
echoLogFile = args.echoLogFile
saveOn = args.saveOn
slurm = args.slurm
propType = args.propType
lowerAlt = args.lowerAlt
upperAlt = args.upperAlt

if echoLogFile:
    gmat.EchoLogFile()


# -----------configuration preliminaries----------------------------


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

# gravity field
earthgrav = gmat.Construct("GravityField")
earthgrav.SetField("BodyName","Earth")
earthgrav.SetField("PotentialFile", 'JGM2.cof')
earthgrav.SetField("Degree",70)
earthgrav.SetField("Order",70)


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

rng = np.random.default_rng(123)
rng.random()

SMA = np.zeros(numRandSys)
ECC = np.zeros(numRandSys)
INC = np.zeros(numRandSys)
RAAN = np.zeros(numRandSys)
AOP = np.zeros(numRandSys)
TA = np.zeros(numRandSys)

for i in range(numRandSys):
    SMA[i] = rng.uniform(R + lowerAlt,R + upperAlt) # km
    ECC[i] = 0.01 * rng.random()
    INC[i] = 10 * rng.random() # deg
    RAAN[i] = 0 # deg
    AOP[i] = 0 # deg
    TA[i] = rng.uniform(-2, 2) # deg

# construct the burn force model
def setThrust(s, b):
    bf = gmat.FiniteThrust("Thrust")
    bf.SetRefObjectName(gmat.SPACECRAFT, s.GetName())
    bf.SetReference(b)
    gmat.ConfigManager.Instance().AddPhysicalModel(bf)
    return bf


if propType == 'chem':
    statesArrayChemical = np.zeros((numRandSys,numMinProp,problemDim))

    tank = gmat.Construct("ChemicalTank", "Fuel") # create a chemical tank with the name "Fuel"
    thruster = gmat.Construct("ChemicalThruster", "Thruster") # create a chemical thruster with the name "Thruster"
    thruster.SetField("C1", chemThrust) # set the thrust coefficient for the "Thruster" to use the chemThrust value
    thruster.SetField("DecrementMass", True)
    thruster.SetField("Tank", "Fuel") # set the tank for the "Thruster" to use the "Fuel" object
    earthorb.SetField("Tanks", "Fuel") # set possible tanks for the "Thruster" to use the "Fuel" object
    earthorb.SetField("Thrusters", "Thruster") # set possible thrusters to use the "Thruster" object

    gmat.Initialize()



    burn = gmat.Construct("FiniteBurn", "TheBurn")
    burn.SetField("Thrusters", "Thruster")
    burn.SetSolarSystem(gmat.GetSolarSystem())
    burn.SetSpacecraftToManeuver(earthorb)

    burnForce = setThrust(earthorb, burn)

    gmat.Initialize()

    print("Generating Chemical Thruster Data...")
    for i in range(numRandSys):

        earthorb.SetField("SMA", SMA[i]) # km
        earthorb.SetField("ECC", ECC[i])
        earthorb.SetField("INC", INC[i]) # deg
        earthorb.SetField("RAAN", RAAN[i]) # deg
        earthorb.SetField("AOP", AOP[i]) # deg
        earthorb.SetField("TA", TA[i]) # deg

        tank.SetField("FuelMass", 200.0)

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
        # theThruster.SetField("IsFiring", True)
        earthorb.IsManeuvering(True)
        burn.SetSpacecraftToManeuver(earthorb)
        # # Add the thrust to the force model
        pdprop.AddForce(burnForce)
        psm = pdprop.GetPropStateManager()
        psm.SetProperty("MassFlow")
        # -----------------------------
        pdprop.PrepareInternals()
        gator = pdprop.GetPropagator()

        for j in range(numMinProp):
            gator.Step(dt)
            elapsed = elapsed + dt
            state = gator.GetState()
            statesArrayChemical[i,j,:] = state[0:6]
            gator.UpdateSpaceObject()

        fm = pdprop.GetODEModel()
        fm.DeleteForce(burnForce)
        # theThruster.SetField("IsFiring", False)
        earthorb.IsManeuvering(False)
        pdprop.PrepareInternals()
        gator = pdprop.GetPropagator()

        del(theThruster)
        del(gator)
        del(fm)
        del(psm)

    print("Chemical Thruster Data Generation Complete.")

    if saveOn:
        if slurm:
            saveDest = '~/pendulumRNN/gmat/data/classification/'
        else:
            saveDest = 'gmat/data/classification/'

        np.savez(saveDest+'statesArrayChemical.npz', statesArrayChemical=statesArrayChemical)

elif propType == 'elec':
    statesArrayElectric = np.zeros((numRandSys,numMinProp,problemDim))


    ETank = gmat.Construct("ElectricTank", "EFuel") # create an electric tank with the name "Fuel"
    EThruster = gmat.Construct("ElectricThruster", "EThruster") # create an electric thruster with the name "Thruster"
    powerSystem = gmat.Construct("SolarPowerSystem", "EPS") # create a power system with the name "EPS"
    EThruster.SetField("ThrustModel", "ThrustMassPolynomial") # set the thrust coefficient for the "EThruster" to use the elecThrust value
    EThruster.SetField("ThrustCoeff1", elecThrust) # set the thrust coefficient for the "EThruster" to use the elecThrust value
    EThruster.SetField("DecrementMass", True)
    EThruster.SetField("ThrustScaleFactor", 0.15) # scale the thrust 
    EThruster.SetField("Tank", "EFuel") # set the tank for the "EThruster" to use the "EFuel" object
    earthorb.SetField("Tanks", "EFuel") # set possible tanks for the "EThruster" to use the "EFuel" object
    earthorb.SetField("Thrusters", "EThruster") # set possible thrusters to use the "EThruster" object
    earthorb.SetField("PowerSystem", "EPS") # set the power system of the spacecraft to use the "EPS" object
    gmat.Initialize()

    # construct the burn force model (the burn force model is the same for both chemical and electric thrusters)
    burn = gmat.Construct("FiniteBurn", "TheEBurn")
    burn.SetField("Thrusters", "EThruster")
    burn.SetSolarSystem(gmat.GetSolarSystem())
    burn.SetSpacecraftToManeuver(earthorb)

    burnForce = setThrust(earthorb, burn)


    gmat.Initialize()

    print("Generating Electric Thruster Data...")
    for i in range(numRandSys):
        earthorb.SetField("SMA", SMA[i]) # km
        earthorb.SetField("ECC", ECC[i])
        earthorb.SetField("INC", INC[i]) # deg
        earthorb.SetField("RAAN", RAAN[i]) # deg
        earthorb.SetField("AOP", AOP[i]) # deg
        earthorb.SetField("TA", TA[i]) # deg

        ETank.SetField("FuelMass", 200.0)

        # Perform initializations
        gmat.Initialize()

        # Refresh the 'gator reference
        gator = pdprop.GetPropagator()

        gmat.Initialize()
        
        pdprop.AddPropObject(earthorb)
        pdprop.PrepareInternals()

        theThruster = earthorb.GetRefObject(gmat.THRUSTER, "EThruster")

        # -----------------------------
        # Finite Burn Specific Settings
        # -----------------------------
        # Turn on the thruster
        # theThruster.SetField("IsFiring", True)
        earthorb.IsManeuvering(True)
        burn.SetSpacecraftToManeuver(earthorb)
        # # Add the thrust to the force model
        pdprop.AddForce(burnForce)
        psm = pdprop.GetPropStateManager()
        psm.SetProperty("MassFlow")
        # -----------------------------
        pdprop.PrepareInternals()
        gator = pdprop.GetPropagator()

        for j in range(numMinProp):
            gator.Step(dt)
            elapsed = elapsed + dt
            state = gator.GetState()
            statesArrayElectric[i,j,:] = state[0:6]
            gator.UpdateSpaceObject()

        fm = pdprop.GetODEModel()
        fm.DeleteForce(burnForce)
        # theThruster.SetField("IsFiring", False)
        earthorb.IsManeuvering(False)
        pdprop.PrepareInternals()
        gator = pdprop.GetPropagator()
        del(theThruster)
        del(gator)
        del(fm)
        del(psm)
    print("Electric Thruster Data Generation Complete.")

    if saveOn:
        if slurm:
            saveDest = '~/pendulumRNN/gmat/data/classification/'
        else:
            saveDest = 'gmat/data/classification/'

        np.savez(saveDest+'statesArrayElectric.npz', statesArrayElectric=statesArrayElectric)

elif propType == 'imp':
    statesArrayImpBurn = np.zeros((numRandSys,numMinProp,problemDim))

    print("Generating Impulsive Burn Data...")
    for i in range(numRandSys):
        earthorb.SetField("SMA", SMA[i]) # km
        earthorb.SetField("ECC", ECC[i])
        earthorb.SetField("INC", INC[i]) # deg
        earthorb.SetField("RAAN", RAAN[i]) # deg
        earthorb.SetField("AOP", AOP[i]) # deg
        earthorb.SetField("TA", TA[i]) # deg

        gmat.Initialize()

        # Refresh the 'gator reference
        gator = pdprop.GetPropagator()

        gmat.Initialize()
        
        pdprop.AddPropObject(earthorb)
        pdprop.PrepareInternals()

        randImpTime = np.random.randint(1, numMinProp-1) # randomly select a time to apply the impulsive burn
        for j in range(numMinProp):
            if j == randImpTime:  
                craftVel = state[3:6]
                craftVel = craftVel/np.linalg.norm(craftVel) * deltaV + craftVel

                earthorb.SetField("VX", craftVel[0])
                earthorb.SetField("VY", craftVel[1])
                earthorb.SetField("VZ", craftVel[2])

                pdprop.PrepareInternals()

            try:
                gator.Step(dt)
            except Exception as e:
                print(f"Error during propagation at i={i}, j={j}: {e}")
                print("Craft State at Error:", state)
                break
            elapsed = elapsed + dt
            state = gator.GetState()
            statesArrayImpBurn[i,j,:] = state[0:6]
            gator.UpdateSpaceObject()

        del(gator)

    print("Impulsive Burn Data Generation Complete.")

    if saveOn:
        if slurm:
            saveDest = '~/pendulumRNN/gmat/data/classification/'
        else:
            saveDest = 'gmat/data/classification/'

        np.savez(saveDest+'statesArrayImpBurn.npz', statesArrayImpBurn=statesArrayImpBurn)

elif propType == 'none':
    statesArrayNoThrust = np.zeros((numRandSys,numMinProp,problemDim))

    print("Generating No Thrust Data...")
    for i in range(numRandSys):
        earthorb.SetField("SMA", SMA[i]) # km
        earthorb.SetField("ECC", ECC[i])
        earthorb.SetField("INC", INC[i]) # deg
        earthorb.SetField("RAAN", RAAN[i]) # deg
        earthorb.SetField("AOP", AOP[i]) # deg
        earthorb.SetField("TA", TA[i]) # deg

        gmat.Initialize()

        # Refresh the 'gator reference
        gator = pdprop.GetPropagator()

        gmat.Initialize()
        
        pdprop.AddPropObject(earthorb)
        pdprop.PrepareInternals()

        for j in range(numMinProp):
            gator.Step(dt)
            elapsed = elapsed + dt
            state = gator.GetState()
            statesArrayNoThrust[i,j,:] = state[0:6]
            gator.UpdateSpaceObject()
        
        del(gator)

    print("No Thrust Data Generation Complete.")

        
    if saveOn:
        if slurm:
            saveDest = '~/pendulumRNN/gmat/data/classification/'
        else:
            saveDest = 'gmat/data/classification/'

        np.savez(saveDest+'statesArrayNoThrust.npz', statesArrayNoThrust=statesArrayNoThrust)
