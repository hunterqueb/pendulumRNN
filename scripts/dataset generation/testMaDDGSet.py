
from pathlib import Path

sensor_yaml = Path("data/SSA/sensor_networks/ex.yaml")


# First, define the maneuver type as a string ("impulse", "continuous", or "all")
mtype = "impulse"

# The vector components of the thrust vector will be pulled from normal distributions, the mean and width of which we define here.
# The values are entered as a 3-element list with the radial, in-track, and cross-track components of the delta-v (respectively)
# in km/s.
dv_ric_mean_kms = (0.0, 0.0, 0.0)
dv_ric_std_kms = (0.0, 0.1, 1.0)


start_mjd = 60197.5
sim_duration_days = 3.0  # Duration is defined in days

num_sim_pairs = 2  # Number of simulation pairs to run

# Create the output directory
output_path = "data/SSA/simulations"
output_dir = Path(output_path)
output_dir.mkdir(exist_ok=True)


multirun_dir = output_dir / "multirun"
multirun_path = str(multirun_dir)  # We need the string version of this path

rm_multirun_root = (
    False  # If True, multirun directories created by this job will be deleted
)


import json

submitit_json = [
    "hydra.job.chdir=True",
    "hydra/launcher=submitit_local",
    "hydra.launcher.nodes=1",
    "hydra.launcher.cpus_per_task=2",
    "hydra.launcher.tasks_per_node=10",
    "hydra.launcher.mem_gb=16",
]

submitit_file = Path("data/SSA/simulations") / "submitit.json"
with open(submitit_file, "w") as f:
    json.dump(submitit_json, f)


from typing import Tuple

import madlib
from maddg._residuals import calculate_residuals
import numpy as np


def simulator_task(
    seq_id: int,
    sensor_params: dict,
    maneuver_type: int,
    sim_duration_days: float,
    start_mjd: float,
    dv_ric_mean_kms: Tuple[float, float, float],
    dv_ric_std_kms=Tuple[float, float, float],
    **kwargs,
):
    # Define a SensorCollection object from the given parameters
    sensors = [madlib.GroundOpticalSensor(**params) for key, params in sensor_params.items()]
    sensor_network = madlib.SensorCollection(sensors)

    # Timing
    epoch = start_mjd

    # Create the satellite (a GEO object at a random longitude)
    sat_longitude = 360 * np.random.random()
    sat_observed = madlib.Satellite.from_GEO_longitude(sat_longitude, epoch)

    maneuver = None
    maneuver_mjd = None
    maneuver_r_kms = None
    maneuver_i_kms = None
    maneuver_c_kms = None

    # For maneuvering cases, create a random maneuver vector
    if maneuver_type == 1:
        # Pick a random maneuver time during the simulation
        maneuver_mjd = epoch + sim_duration_days * np.random.random()

        # Calculate the thrust vector using the input distributions
        mean_rad, mean_in, mean_crs = dv_ric_mean_kms
        std_rad, std_in, std_crs = dv_ric_std_kms

        maneuver_r_kms = mean_rad + std_rad * np.random.randn()
        maneuver_i_kms = mean_in + std_in * np.random.randn()
        maneuver_c_kms = mean_crs + std_crs * np.random.randn()

        # Define the ImpulsiveManeuver object, converting from km/s to m/s
        man_dv = np.array([maneuver_r_kms, maneuver_i_kms, maneuver_c_kms]) / 1000
        maneuver = madlib.ImpulsiveManeuver(maneuver_mjd, man_dv)

    sat_observed.maneuver = maneuver

    # Observe and calculate residuals
    residual_df = calculate_residuals(
        sensors=sensor_network,
        satellite=sat_observed,
        sim_duration_days=sim_duration_days,
        t_start_mjd=epoch,
    )

    # Append maneuver information to the output dataframe
    if residual_df is not None:
        residual_df["Maneuver"] = maneuver_type
        residual_df["Sequence"] = int(seq_id)
        residual_df["Maneuver_MJD"] = maneuver_mjd
        residual_df["Maneuver_DV_Radial_KmS"] = maneuver_r_kms
        residual_df["Maneuver_DV_InTrack_KmS"] = maneuver_i_kms
        residual_df["Maneuver_DV_CrossTrack_KmS"] = maneuver_c_kms

    # Return the requisite dataframe
    return residual_df

from maddg._sim_launcher import launcher

launcher(
    simulator_method=simulator_task,
    mtype=mtype,
    num_sim_pairs=num_sim_pairs,
    sensor_yaml=sensor_yaml,
    outdir=output_dir,
    dv_ric_mean_kms=dv_ric_mean_kms,
    dv_ric_std_kms=dv_ric_std_kms,
    submitit=str(submitit_file),
    multirun_root=multirun_path,
    rm_multirun_root=rm_multirun_root,
    start_mjd=start_mjd,
    sim_duration_days=sim_duration_days,
    random_seed=0,
)
