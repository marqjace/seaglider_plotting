#!/usr/bin/env python
# coding: utf-8
# Created by J. Marquardt on 2/8/2024
# Last edited on 2/8/2024
#    Need to add climb to T-S diagram and swap sigma-t colorbar for depth or time
#    Need to add lat-lon map

# Imports
import os
import math
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RBFInterpolator

# Filename for dive data
filename = r'C:/Users/marqjace/TH_line/deployments/oct_2024/transect2/p2660111.nc'

# Output folder for plots
output_folder = r'C:/Users/marqjace/TH_line/deployments/oct_2024/transect2/dive_111/'
os.makedirs(output_folder, exist_ok=True)  # exist_ok=True prevents errors if the folder already exists

# Function to split seaglider dive data into ascent and descent phases
def split_sg_profile(ds, threshold=0.07):
    """
    Splits seaglider dive data into ascent and descent phases.
    
    Parameters:
        ds (xr.Dataset): Dataset containing at least 'time' and 'pressure' variables.
        
    Returns:
        dive (xr.Dataset): Dataset containing descent data.
        climb (xr.Dataset): Dataset containing ascent data.
    """
    
    # Ensure the data is sorted by time
    ds = ds.sortby('ctd_time')

    # Convert ctd_time to numerical format (Unix epoch time in seconds)
    ds = ds.assign_coords(
        ctd_time=(ds["ctd_time"] - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    )

    # Now ctd_time is in seconds, and you can safely apply differentiate()
    depth_diff = ds['ctd_depth'].differentiate('ctd_time')
    
    # Identify ascent and descent using the threshold
    dive = ds.where(depth_diff > threshold, drop=True)  # Descent: pressure increase greater than threshold
    climb = ds.where(depth_diff < -threshold, drop=True)  # Ascent: pressure decrease greater than threshold
    
    return dive, climb

# Open the dataset
ds = xr.open_dataset(filename)

# Split the dataset into dive and climb profiles
dive, climb = split_sg_profile(ds)

# Function to plot seaglider dive and climb profiles
def sg_plots(dive_ds, climb_ds):
    """
    Plots seaglider dive and climb profiles.
    
    Parameters:
        dive_ds (xr.Dataset): Split dive dataset.
        climb_ds (xr.Dataset): Split climb dataset.
        
    Returns:
        dive plots (.png): 
                    Depth Profile,
                    Temperature Profile,
                    Salinity Profile,
                    Conductivity Profile,
                    T-S Diagram,
                    Density Profile,
                    Sigma-t Profile,
                    Pressure Profile,
                    Lat-lon Map,
                    Buoyancy Profile,
                    Heading Profile,
                    Vertical Speed Profile,
                    Glide Angle Profile,

        climb plots (.png): 
                    Depth Profile,
                    Temperature Profile,
                    Salinity Profile,
                    Conductivity Profile,
                    T-S Diagram,
                    Density Profile,
                    Sigma-t Profile,
                    Pressure Profile,
                    Lat-lon Map,
                    Buoyancy Profile,
                    Heading Profile,
                    Vertical Speed Profile,
                    Glide Angle Profile,
    """
    
    # Depth Profile
    fig, ax = plt.subplots(figsize=(8,8), dpi=300)
    dive_plot = ax.plot(dive.ctd_time, dive.ctd_depth, label='Dive', c='r')
    climb_plot = ax.plot(climb.ctd_time, climb.ctd_depth, label='Climb', c='b')
    ax.invert_yaxis()
    ax.set_xlabel('Time (seconds since 1970-01-01)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Depth Profiles')
    ax.legend()
    ax.grid()
    fig.savefig(output_folder + 'depth_profiles.png')
    
    # Temperature Profile
    fig, ax = plt.subplots(figsize=(8,8), dpi=300)
    dive_plot = ax.plot(dive.temperature, dive.ctd_depth, label='Dive', c='r')
    climb_plot = ax.plot(climb.temperature, climb.ctd_depth, label='Climb', c='b')
    ax.invert_yaxis()
    ax.set_xlabel(f'Temperature ($\degree$C)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Temperature Profiles')
    ax.legend()
    ax.grid()
    fig.savefig(output_folder + 'temp_profiles.png')

    # Salinity Profile
    fig, ax = plt.subplots(figsize=(8,8), dpi=300)
    dive_plot = ax.plot(dive.salinity, dive.ctd_depth, label='Dive', c='r')
    climb_plot = ax.plot(climb.salinity, climb.ctd_depth, label='Climb', c='b')
    ax.invert_yaxis()
    ax.set_xlabel('Salinity (PSU)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Salinity Profiles')
    ax.legend()
    ax.grid()
    fig.savefig(output_folder + 'salinity_profiles.png')

    # Conductivity Profile
    fig, ax = plt.subplots(figsize=(8,8), dpi=300)
    dive_plot = ax.plot(dive.conductivity, dive.ctd_depth, label='Dive', c='r')
    climb_plot = ax.plot(climb.conductivity, climb.ctd_depth, label='Climb', c='b')
    ax.invert_yaxis()
    ax.set_xlabel('Conductivity (S/m)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Conductivity Profiles')
    ax.legend()
    ax.grid()
    fig.savefig(output_folder + 'conductivity_profiles.png')

    # Density Profile
    fig, ax = plt.subplots(figsize=(8,8), dpi=300)
    dive_plot = ax.plot(dive.density, dive.ctd_depth, label='Dive', c='r')
    climb_plot = ax.plot(climb.density, climb.ctd_depth, label='Climb', c='b')
    ax.invert_yaxis()
    ax.set_xlabel(f'Density (g/$m^3$)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Density Profiles')
    ax.legend()
    ax.grid()
    fig.savefig(output_folder + 'density_profiles.png')

    # T-S Plot
    # Define min/max values with an extended range
    t_min, t_max = dive.temperature.min().values, dive.temperature.max().values
    s_min, s_max = dive.salinity.min().values, dive.salinity.max().values

    # Linearly spaced temperature and salinity ranges
    tempL = np.linspace(t_min - 1, t_max + 1)  # Extended temperature range
    salL = np.linspace(s_min - .5, s_max + .5)  # Extended salinity range
    
    # Create a meshgrid for the temperature and salinity
    Tg, Sg = np.meshgrid(tempL, salL)

    # Flatten the original data to use with NearestNDInterpolator
    points = np.column_stack((dive.salinity.values, dive.temperature.values))
    values = dive.sigma_t.values

    # Use RBF interpolation (which allows extrapolation)
    interp_func = RBFInterpolator(points, values, smoothing=0.1)  # Adjust smoothing if needed

    # Evaluate interpolation over the full grid
    grid_points = np.column_stack((Sg.ravel(), Tg.ravel()))
    sigma_theta_grid = interp_func(grid_points).reshape(Sg.shape)

    # Generate contour levels
    cnt = np.linspace(np.min(sigma_theta_grid), np.max(sigma_theta_grid), 10)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Contour lines
    cl = ax.contour(Sg, Tg, sigma_theta_grid, levels=cnt, colors='black', linewidths=0.5)
    plt.clabel(cl, fontsize=8, fmt="%.1f")

    # Scatter plot for original data points
    sc = ax.scatter(dive.salinity, dive.temperature, c=dive.sigma_t, s=10)

    # Colorbar
    cb = plt.colorbar(sc, label=f'Sigma-t (g/$m^3$)')

    ax.set_xlabel('Salinity (PSU)')
    ax.set_ylabel('Temperature [$\degree$C]')
    ax.set_title('T-S Diagram', fontsize=14, fontweight='bold')
    fig.savefig(output_folder + 'TS_plot.png')


    # Sigma-t Profile 
    fig, ax = plt.subplots(figsize=(8,8), dpi=300)
    dive_plot = ax.plot(dive.sigma_t, dive.ctd_depth, label='Dive', c='r')
    climb_plot = ax.plot(climb.sigma_t, climb.ctd_depth, label='Climb', c='b')
    ax.invert_yaxis()
    ax.set_xlabel(f'Sigma-t ($g/m^3$)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Sigma-t Profiles')
    ax.legend()
    ax.grid()
    fig.savefig(output_folder + 'sigma-t_profiles.png')   

    # Pressure Profile 
    fig, ax = plt.subplots(figsize=(8,8), dpi=300)
    dive_plot = ax.plot(dive.ctd_time, dive.ctd_pressure, label='Dive', c='r')
    climb_plot = ax.plot(climb.ctd_time, climb.ctd_pressure, label='Climb', c='b')
    ax.invert_yaxis()
    ax.set_ylabel(f'Pressure (dbar)')
    ax.set_xlabel('Time (seconds since 1970-01-01)')
    ax.set_title('Pressure Profiles')
    ax.legend()
    ax.grid()
    fig.savefig(output_folder + 'pressure_profiles.png')   

    # Buoyancy Profile 
    fig, ax = plt.subplots(figsize=(8,8), dpi=300)
    dive_plot = ax.plot(dive.buoyancy, dive.ctd_depth, label='Dive', c='r')
    climb_plot = ax.plot(climb.buoyancy, climb.ctd_depth, label='Climb', c='b')
    ax.invert_yaxis()
    ax.set_xlabel(f'Buoyancy (g)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Buoyancy Profiles')
    ax.legend()
    ax.grid()
    fig.savefig(output_folder + 'buoyancy_profiles.png')

    # Polar Heading vs time 
    # Convert from radians E to degrees N and ensure the range is 0-360
    dive_heading_N = (90 - dive.polar_heading * 180 / math.pi) % 360
    climb_heading_N = (90 - climb.polar_heading * 180 / math.pi) % 360
    
    fig, ax = plt.subplots(figsize=(8,8), dpi=300)
    dive_plot = ax.plot(dive_heading_N, dive.ctd_depth, label='Dive', c='r')
    climb_plot = ax.plot(climb_heading_N, climb.ctd_depth, label='Climb', c='b')
    ax.invert_yaxis()
    ax.set_xlabel(f'Heading ($\degree$N)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Heading Profiles')
    ax.legend()
    ax.grid()
    fig.savefig(output_folder + 'heading_profiles.png') 

    # Vertical Speed vs time     
    fig, ax = plt.subplots(figsize=(8,8), dpi=300)
    dive_plot = ax.plot(dive.vert_speed, dive.ctd_depth, label='Dive', c='r')
    climb_plot = ax.plot(climb.vert_speed, climb.ctd_depth, label='Climb', c='b')
    ax.invert_yaxis()
    ax.set_xlabel(f'Vertical Speed (cm/s)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Vertical Speed Profiles')
    ax.legend()
    ax.grid()
    fig.savefig(output_folder + 'vert_speed_profiles.png')

    # Glide Angle vs time     
    fig, ax = plt.subplots(figsize=(8,8), dpi=300)
    dive_plot = ax.plot(dive.glide_angle, dive.ctd_depth, label='Dive', c='r')
    climb_plot = ax.plot(climb.glide_angle, climb.ctd_depth, label='Climb', c='b')
    ax.invert_yaxis()
    ax.set_xlabel(f'Glide Angle (degrees)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Glide Angle Profiles')
    ax.legend()
    ax.grid()
    fig.savefig(output_folder + 'glide_angle_profiles.png') 


sg_plots(dive, climb)