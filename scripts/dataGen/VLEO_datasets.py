import numpy as np
import os
from qutils.integrators import ode45
from qutils.orbital import OE2ECI
from qutils.tictoc import timer


def twoBodyJ2Drag(t, y, mu,m_sat):
    # two body problem with J2 perturbation in 6 dimensions taken from astroforge library
    # https://github.com/mit-ll/AstroForge/blob/main/src/astroforge/force_models/_models.py
    # https://github.com/mit-ll/AstroForge/blob/main/src/astroforge/force_models/_forces.py

    # x, v = np.split(y, 2) # orginal line in Astroforge
    # faster than above
    x = y[:3]
    v = y[3:]

    J2 = 4.84165368e-4 * np.sqrt(5)

    M2 = J2 * np.diag(np.array([0.5, 0.5, -1.0]))
    r = np.sqrt(x @ x) # faster than np.linalg.norm(x) (original line in Astroforge)
    v_norm = np.sqrt(v @ v) # faster than np.linalg.norm(v)

    # compute monopole force
    F0 = -mu * x / r**3

    # compute the quadropole force in ITRS
    acc = (mu * R**2 / r**5) * (-5 * x * (x @ M2 @ x) / r**2 + 2 * M2 @ x) + F0

    # ydot = np.hstack((v, acc)) # orginal line in Astroforge
    # faster than above
    ydot = np.empty(6)
    ydot[:3] = v
    ydot[3:] = acc

    rho = rho_0 * np.exp(-(r-R) / h_scale)  # Atmospheric density model
    drag_factor = -0.5 * (rho / m_sat) * c_d * A_sat * v_norm

    a_drag = v * drag_factor
    ydot[3:] += a_drag

    # print(f"rho: {rho}, satellite mass: {m_sat}, a_drag: {a_drag}, force: {np.linalg.norm(ydot[3:]*m_sat)}")

    return ydot

import argparse

parser = argparse.ArgumentParser(description='Generate VLEO datasets for classification tasks')
parser.add_argument('--numRandSys', type=int, default=10000, help='Number of random systems to generate')
parser.add_argument('--mass', type=int, default=100, help='Maximum mass for classification')
parser.add_argument('--orbits',type=int,default=1,help="Number of Orbits for Propagation")
parser.add_argument('--timeSeriesLength',type=int,default=1000,help="Number of time steps in the time series")
parser.add_argument('--e',type=float,default=0,help="Max eccentricity of the Orbit Regimes")
parser.add_argument('--i',type=float,default=0,help="Max inclination of the orbit regimes in degrees")
parser.add_argument('--out', default='vleo_dataset', help='Base name of output file')
parser.add_argument('--folder', default='data/massClassification/VLEO/', help='Output folder for the dataset')

args = parser.parse_args()
numRandSys = args.numRandSys
mass_max = args.mass
numOrbits = args.orbits
timeSeriesLength = args.timeSeriesLength
e_max = args.e
inc_max = args.i
filename = args.out
folder = args.folder

filepath = folder + filename

print(f"Maximum mass for classification : {mass_max * 2} kg")
print(f"Number of random systems        : {numRandSys}")
print(f"Number of Orbits to Propagate   : {numOrbits}")
print(f"Max Eccentricity of Orbits      : {e_max}")
print(f"Max Inclination of Orbits in deg: {inc_max}")
print(f"Length of Time Series           : {timeSeriesLength}")


rng = np.random.default_rng() # Seed for reproducibility

# Hyperparameters
problemDim = 6
input_size = problemDim   # 2

# Orbital Parameters
G = 6.67430e-11 # m^3/kg/s^2, gravitational constant
M_earth = 5.97219e24 # kg, mass of Earth
mu = 3.986004418e14  # Earth’s mu in m^3/s^2
R = 6371e3 # radius of earth in m

DU = R
TU = np.sqrt(R**3/mu) # time unit in seconds
A_sat = 10 # m^2, cross section area of satellite

# Atmospheric model parameters
rho_0 = 1.29 # kg/m^3
c_d = 2.1 #shperical model
h_scale = 5000


timeToGenData = timer()

numericalResult = np.zeros((numRandSys,timeSeriesLength,problemDim))
numericalResultTime = np.zeros((numRandSys,timeSeriesLength,1))
mass_array = np.zeros((numRandSys,))

# define the semimajor axis as a function of the random number generator if constant_a is False
# otherwise, use a constant value

for i in range(numRandSys):
    # Random Conditions for dataset generation
    m_sat = mass_max * rng.random() + mass_max # mass of satellite in kg
    e = e_max * rng.random() # eccentricity
    inc = np.deg2rad(inc_max * rng.random()) # inclination
    a = rng.uniform(R + 100e3,R + 200e3) # random semimajor axis in m
    nu = np.deg2rad(5*rng.random()) # true anomaly

    # calc mu from mass of satellite and earth to get better accuracy(??)
    mu = G * (M_earth + m_sat) # gravitational parameter in m^3/s^2

    h = np.sqrt(mu*a*(1-e)) # specific angular momentum

    OE = [a,e,inc,0,0,nu]
    y0 = OE2ECI(OE,mu=mu)
    # print(y0)

    tf = 2*np.pi*a**2*np.sqrt(1-e**2)/h * numOrbits # time of flight

    teval = np.linspace(0, tf, timeSeriesLength) # time to evaluate the solution

    t,y = ode45(fun=lambda t, y: twoBodyJ2Drag(t, y, mu,m_sat),tspan=(0, tf),y0=y0, t_eval=teval, rtol=1e-8, atol=1e-10)

    # save the time series data
    numericalResult[i,:,:] = y
    numericalResultTime[i,:,:] = t
    mass_array[i] = m_sat

print("Time to generate data: {:.2f} seconds".format(timeToGenData.tocVal()))

run_info = {
    'numRandSys': numRandSys,
    'mass_max': mass_max,
    'numOrbits': numOrbits,
    'timeSeriesLength': timeSeriesLength,
    'e_max': e_max,
    'inc_max': inc_max,
    'rho_0': rho_0,
    'c_d': c_d,
    'h_scale': h_scale,
    'A_sat': A_sat
}
np.savez_compressed(
    f'{filepath}.npz',
    numericalResult  = numericalResult.astype(np.float32),     # shape [N, T, 6]
    numericalResultTime  = numericalResultTime.astype(np.float32), # shape [N, T, 1]
    mass_array  = mass_array.astype(np.float32),          # shape [N]
    meta = np.array([run_info], dtype=object)    # pickle the dict
)

print(f'Saved dataset → {filepath}.npz  ({os.path.getsize(filepath + ".npz")/1e6:.2f} MB)')

# loading the dataset
def load_npz(fname):
    d = np.load(fname, allow_pickle=True)
    return d['X'], d['t'], d['m']

