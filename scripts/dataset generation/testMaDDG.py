import madlib
import numpy as np
from qutils.tictoc import timer
# following the example in the documentation

# intial epoch of satellite in modified julian date
epoch_mjd = 60197.5

# initial position and velocity of the satellite in geocentric coordinates
leo_alt_km = 430.0
leo_speed_kms = 27500.0 / 3600
earth_radius_km = 6378.0

# position and velocity of the satellite in geocentric coordinates
leo_pos_teted = np.array([leo_alt_km + earth_radius_km, 0.0, 0.0])
leo_vel_teted = leo_speed_kms * np.array([0.0, 2**-0.5, 2**-0.5])

# intilize the satellite object
leo_satellite = madlib.Satellite(epoch=epoch_mjd, pos=leo_pos_teted, vel=leo_vel_teted)

geo_satellite_0 = madlib.Satellite.from_GEO_longitude(lon = 0, epoch=epoch_mjd)

geo_satellite_270 = madlib.Satellite.from_GEO_longitude(lon = 270, epoch=epoch_mjd)


# defining the sensor, based on the haystack observatory
# Defines a sensor "Needle" located at the Haystack Observatory in Westford, MA
# at latitude 42.6233, longitude -71.4882, and altitude 0.131 km,
# with an imaginary telescope with astrometric measurements accurate to within ten arcseconds
# collecting 3 observations per collection, with a 1 second spacing between observations,
# with a mean collection gap of 60 seconds and a standard deviation of 5 seconds.

sensor_params = {
    "id" : "Needle",
    "lat" : 42.6233,
    "lon" : -71.4882,
    "alt" : 0.131,
    "dra": 10,
    "ddec": 10,
    "obs_per_collect" : 3,
    "obs_time_spacing" : 1,
    "collect_gap_mean": 60,
    "collect_gap_std": 5
    }

# observation limits
# limiting behaviors that restric the observations of the sensor, such as a sensor that cannot
# point below a ecrtain elevation angle, cam only observe a certain range of azimuth angles, 
# or is sensitive to background brightness in early twilight, etc.

# the example defines sensor that has a minimum pointing elevation of 15 deg, and is only effective during astronmical twilight
# (sun is 18 or more degres below the horizon)

obs_limits = {
    "el" : (15,90),
    "sun_el" : (-90, -18)
}

sensor_params["obs_limits"] = obs_limits

# create the sensor object
sensor = madlib.GroundOpticalSensor(**sensor_params)

start_mjd = epoch_mjd
end_mjd = epoch_mjd + 3 # 3 days of observations

geo_observations_0 = sensor.observe(target_satellite=geo_satellite_0, times=(start_mjd, end_mjd))

print("Number of valid observations for GEO 0", geo_observations_0.count_valid_observations())


geo_observations_270 = sensor.observe(target_satellite=geo_satellite_270, times=(start_mjd, end_mjd))

print("Number of valid observations for GEO 270", geo_observations_270.count_valid_observations())

leoTimeToObserve=timer()
leo_observations = sensor.observe(target_satellite=leo_satellite, times=(start_mjd, end_mjd))
leoTimeToObserve.toc()

print("Number of valid observations for LEO", leo_observations.count_valid_observations())

from matplotlib import pyplot as plt

times = np.array([obs.mjd for obs in geo_observations_270.pos_observed])
time = 24 * (times - min(times))  # convert to hours

ra = np.array([obs.ra for obs in geo_observations_270.pos_observed])
dec = np.array([obs.dec for obs in geo_observations_270.pos_observed])

fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(9, 9))

_ = ax1.plot(times, ra, "ob", ms=5, lw=2)
ax1.set_ylabel("Right Ascension (degrees)", fontsize=16)
_ = ax2.plot(times, dec, "ob", ms=5, lw=2)
ax2.set_ylabel("Declination (degrees)", fontsize=16)
ax2.set_xlabel("Time (Hours)", fontsize=16)
ax1.grid()
ax2.grid()


times = np.array([o.mjd for o in leo_observations.pos_observed])
times = 24 * (times - min(times))  # Convert the times from days to hours

ra = [o.ra for o in leo_observations.pos_observed]
dec = [o.dec for o in leo_observations.pos_observed]

fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(9, 9))

_ = ax1.plot(times, ra, "ob", ms=5, lw=2)
ax1.set_ylabel("Right Ascension (degrees)", fontsize=16)
_ = ax2.plot(times, dec, "ob", ms=5, lw=2)
ax2.set_ylabel("Declination (degrees)", fontsize=16)
ax2.set_xlabel("Time (Hours)", fontsize=16)
ax1.grid()
ax2.grid()

plt.show()
