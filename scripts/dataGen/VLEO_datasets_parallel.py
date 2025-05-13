import numpy as np
import os

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

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


rng = np.random.default_rng() # Seed for reproducibility

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

def simulate_one(idx, *, args, seed_base=12345):
    rng = np.random.default_rng(seed_base + idx)   # deterministic per‑task RNG

    # ----- draw one random orbit ------------------
    m_sat = args.mass * rng.random() + args.mass
    e     = args.e * rng.random()
    inc   = np.deg2rad(args.i * rng.random())
    a     = rng.uniform(R + 100e3, R + 200e3)
    nu    = np.deg2rad(5 * rng.random())

    mu_local = G * (M_earth + m_sat)
    h        = np.sqrt(mu_local * a * (1 - e))
    OE       = [a, e, inc, 0, 0, nu]
    y0       = OE2ECI(OE, mu=mu_local)

    tf       = 2 * np.pi * a**2 * np.sqrt(1 - e**2) / h * args.orbits
    teval    = np.linspace(0, tf, args.timeSeriesLength)

    t, y = ode45(lambda t, y: twoBodyJ2Drag(t, y, mu_local, m_sat),
                 tspan=(0, tf), y0=y0, t_eval=teval,
                 rtol=1e-8, atol=1e-10)

    return idx, y.astype(np.float32), t.astype(np.float32), np.float32(m_sat)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a VLEO dataset with parallel numerical integrations.")
    parser.add_argument('--numRandSys', type=int, default=10_000,
                        help='Number of random trajectories to generate')
    parser.add_argument('--mass', type=float, default=100.0,
                        help='Reference satellite mass (kg)')
    parser.add_argument('--orbits', type=int, default=1,
                        help='Number of orbits per integration')
    parser.add_argument('--timeSeriesLength', type=int, default=1000,
                        help='Number of evaluation points in each trajectory')
    parser.add_argument('--e', type=float, default=0.0,
                        help='Maximum eccentricity (uniform [0,e])')
    parser.add_argument('--i', type=float, default=0.0,
                        help='Maximum inclination in degrees (uniform [0,i])')
    parser.add_argument('--out', default='vleo_dataset',
                        help='Base name of the output file (without extension)')
    parser.add_argument('--folder', default='data/massClassification/VLEO/', help='Output folder for the dataset')
    parser.add_argument('--workers', type=int, default=mp.cpu_count(),
                        help='Number of parallel processes (default: all CPUs)')
    args = parser.parse_args()
    numRandSys = args.numRandSys
    mass_max = args.mass
    numOrbits = args.orbits
    timeSeriesLength = args.timeSeriesLength
    e_max = args.e
    inc_max = args.i
    filename = args.out
    folder = args.folder

    print(f"Maximum mass for classification : {mass_max * 2} kg")
    print(f"Number of random systems        : {numRandSys}")
    print(f"Number of Orbits to Propagate   : {numOrbits}")
    print(f"Max Eccentricity of Orbits      : {e_max}")
    print(f"Max Inclination of Orbits in deg: {inc_max}")
    print(f"Length of Time Series           : {timeSeriesLength}")

    # Pre‑allocate output arrays (float32 to cut memory in half)
    N = args.numRandSys
    T = args.timeSeriesLength
    numericalResult  = np.empty((N, T, 6), dtype=np.float32)  # state vectors
    numericalResultTime = np.empty((N, T, 1), dtype=np.float32)  # timestamps
    mass_array  = np.empty((N,),      dtype=np.float32)  # satellite masses

    print(f"\nGenerating {N:,} trajectories | workers={args.workers}\n")
    tic = timer()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(simulate_one, i, args=args) for i in range(N)]
        for fut in tqdm(as_completed(futures), total=N, desc="Integrating", ncols=80):
            idx, y_out, t_out, m_sat = fut.result()
            numericalResult[idx]  = y_out
            numericalResultTime[idx] = t_out
            mass_array[idx]  = m_sat

    elapsed = tic.tocVal()
    print(f"Done in {elapsed:.1f} s ≈ {elapsed/N:.3f} s per orbit")

    # Save dataset (compressed .npz) + provenance
    meta = vars(args)  # convert Namespace → dict
    folder = args.folder

    filepath = folder + filename

    np.savez_compressed(
        f"{filepath}.npz",
        numericalResult=numericalResult,
        numericalResultTime=numericalResultTime,
        mass_array=mass_array,
        meta=np.array([meta], dtype=object),  # pickle the dict
    )
    size_mb = os.path.getsize(f"{filepath}.npz") / 1e6
    print(f"\nSaved → {filepath}.npz  ({size_mb:.2f} MB)\n")


# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Use spawn so that Windows & Jupyter behave well with multiprocessing
    mp.set_start_method("spawn", force=True)
    main()

# loading the dataset
def load_npz(fname):
    d = np.load(fname, allow_pickle=True)
    return d['X'], d['t'], d['m']

