# Takes indiviual dive files, splits them into dive and climb profiles, and makes some plots.
# Author: Jace Marquardt
# Last Updated: 2025-02-11

# Imports
import os
import math
import glob
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RBFInterpolator
from split_sg_profile import split_sg_profile

# Open the filepath where the data is located
filepath = r'C:/Users/marqjace/TH_line/deployments/oct_2024/transect1/'
os.makedirs(filepath, exist_ok=True)  # exist_ok=True prevents errors if the folder already exists

# Specify which data files you want to open
data_files = r'p2660*.nc'
files = glob.glob(filepath + data_files)

# Defines a 'figures' directory to store the output figures
figures = r'figures/'

# Function to plot seaglider dive and climb profiles
def sg_plots(files):
    """
    Plots seaglider dive and climb profiles.
    
    Parameters:
        files (list): List of dive files.
        
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

    datasets = {f'dive_{idx}': xr.open_dataset(file) for idx, file in enumerate(files)}

    # Create plots for each dive
    for idx, ds in datasets.items():

        # Split the dataset into dive and climb profiles
        dive, climb = split_sg_profile(ds)
        
        # Define dive number
        dive_num = dive.trajectory.values

        # Create a dictionary with the science variables on the dive
        science_vars_dive = {
            'Temperature': dive.temperature,
            'Salinity': dive.salinity,
            'Conductivity': dive.conductivity,
            'Density': dive.density,
            'Sigma_theta': dive.sigma_t,
            'Buoyancy': dive.buoyancy,
            'Vertical Speed': dive.vert_speed,
            'Glide Angle': dive.glide_angle,
        }

        science_vars_climb = {
            'Temperature': climb.temperature,
            'Salinity': climb.salinity,
            'Conductivity': climb.conductivity,
            'Density': climb.density,
            'Sigma_theta': climb.sigma_t,
            'Buoyancy': climb.buoyancy,
            'Vertical Speed': climb.vert_speed,
            'Glide Angle': climb.glide_angle,
        }

        # Create output folder
        output_folder = filepath + figures + f'dive_{dive_num}/'
        os.makedirs(output_folder, exist_ok=True)  # exist_ok=True prevents errors if the folder already exists

        # Plot depth vs time
        fig, ax = plt.subplots(figsize=(8,8), dpi=300)
        ax.plot(dive.ctd_time, dive.ctd_depth, c='r', label='Dive')
        ax.plot(climb.ctd_time, climb.ctd_depth, c='b', label='Climb')
        ax.invert_yaxis()
        ax.set_xlabel('Time (seconds since 1970-01-01)')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'Dive {dive_num} - Depth vs Time')
        ax.legend()
        ax.grid()
        ax.set_facecolor('0.9')
        fig.savefig(output_folder + f'depth.png')
        plt.close()
        
        # Pressure Profile 
        fig, ax = plt.subplots(figsize=(8,8), dpi=300)
        ax.plot(dive.ctd_time, dive.ctd_pressure, label='Dive', c='r')
        ax.plot(climb.ctd_time, climb.ctd_pressure, label='Climb', c='b')
        ax.invert_yaxis()
        ax.set_ylabel(f'Pressure (dbar)')
        ax.set_xlabel('Time (seconds since 1970-01-01)')
        ax.set_title('Pressure Profiles')
        ax.legend()
        ax.grid()
        ax.set_facecolor('0.9')
        fig.savefig(output_folder + 'pressure.png')
        plt.close()

        # Polar Heading vs time 
        # Convert from radians E to degrees N and ensure the range is 0-360
        dive_heading_N = (90 - dive.polar_heading * 180 / math.pi) % 360
        climb_heading_N = (90 - climb.polar_heading * 180 / math.pi) % 360
        
        fig, ax = plt.subplots(figsize=(8,8), dpi=300)
        ax.plot(dive_heading_N, dive.ctd_depth, label='Dive', c='r')
        ax.plot(climb_heading_N, climb.ctd_depth, label='Climb', c='b')
        ax.invert_yaxis()
        ax.set_xlabel(f'Heading ($\degree$N)')
        ax.set_ylabel('Depth (m)')
        ax.set_title('Heading Profiles')
        ax.legend()
        ax.grid()
        ax.set_facecolor('0.9')
        fig.savefig(output_folder + 'heading_profiles.png') 
        plt.close()

        # T-S Plot
        # Define min/max values with an extended range
        t_min, t_max = dive.temperature.min().values, dive.temperature.max().values
        s_min, s_max = dive.salinity.min().values, dive.salinity.max().values
        tempL = np.linspace(t_min - .5, t_max + .5)  # Extended temperature range
        salL = np.linspace(s_min - .25, s_max + .25)  # Extended salinity range
        Tg, Sg = np.meshgrid(tempL, salL)

        # Flatten the original data to use with RBFInterpolator
        points = np.column_stack((dive.salinity.values, dive.temperature.values))
        values = dive.sigma_t.values
        interp_func = RBFInterpolator(points, values, smoothing=0.1)  # Adjust smoothing if needed

        # Evaluate interpolation over the full grid
        grid_points = np.column_stack((Sg.ravel(), Tg.ravel()))
        sigma_theta_grid = interp_func(grid_points).reshape(Sg.shape)

        # Generate contour levels
        cnt = np.linspace(np.min(sigma_theta_grid), np.max(sigma_theta_grid), 10)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        cl = ax.contour(Sg, Tg, sigma_theta_grid, levels=cnt, colors='black', linewidths=0.5)
        plt.clabel(cl, fontsize=8, fmt="%.1f")
        sc = ax.scatter(dive.salinity, dive.temperature, c=dive.sigma_t, s=10, cmap='jet_r')
        cb = plt.colorbar(sc, label=f'Sigma-t (g/$m^3$)')
        ax.set_xlabel('Salinity (PSU)')
        ax.set_ylabel('Temperature [$\degree$C]')
        ax.set_title('T-S Diagram', fontsize=14, fontweight='bold')
        ax.set_facecolor('0.9')
        fig.savefig(output_folder + 'TS_plot.png')
        plt.close()

        # Plot science variables vs depth
        for var_dive, var_climb in zip(science_vars_dive, science_vars_climb):
            fig, ax = plt.subplots(figsize=(8,8), dpi=300)
            ax.plot(science_vars_dive[var_dive].values, dive.ctd_depth, c='r', label='Dive')
            ax.plot(science_vars_climb[var_climb].values, climb.ctd_depth, c='b', label='Climb')
            units = science_vars_dive[var_dive].attrs.get("units", "No units specified")
            ax.invert_yaxis()
            ax.set_xlabel(f'{var_dive} ({units})')
            ax.set_ylabel('Depth (m)')
            ax.set_title(f'Dive {dive_num} - {var_dive} vs Depth')
            ax.legend()
            ax.grid()
            ax.set_facecolor('0.9')
            fig.savefig(output_folder + f'{var_dive}.png')
            plt.close()

    for ds in datasets.values():
        ds.close()

sg_plots(files)
