import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import os

# Load the timeseries data
filepath = r"C:\Users\marqjace\seaglider\sg266\sg266_2024_10_21_TH_Line_timeseries.nc"
figures_folder = r"C:\Users\marqjace\seaglider\sg266\figures"
os.makedirs(figures_folder, exist_ok=True)  # exist_ok=True prevents errors if the folder already exists

def sg_timeseries(filepath, figures_folder):
    """
    Creates directories of each of the science variables and plots several timeseries figures.
    
    Parameters:
        filepath: Filepath to the timeseries data.
        figures_folder: Folder where you want the figures to go (will create folder if it does not exist.
        
    Returns:
        directories: Directories containing the following figures of each of the specified science variables:
                     - Raw scatter plot
                     - Gridded scatter plot (interpolated by [20m x 3days]
                     - Gridded contour plot (interpolated by [20m x 3days]
    """
    # Open the dataset
    ds = xr.open_dataset(filepath)
    
    # Define variables
    time_coverage_start = ds.attrs['time_coverage_start']
    time_coverage_end = ds.attrs['time_coverage_end']
    ctd_time = ds.ctd_time
    ctd_depth = ds.ctd_depth
    temperature = ds.temperature
    salinity = ds.salinity
    oxygen = ds.aanderaa4831_dissolved_oxygen

    # Define the science variables
    sci_variables = {
        'temperature': temperature,
        'salinity': salinity,
        'oxygen':oxygen
    }

    # Convert ctd_time  to datetime and as int64
    ctd_time_dt = pd.to_datetime(ctd_time.values, unit='s', origin='unix')
    ctd_time_timestamps = ctd_time_dt.astype(np.int64) // 10**9

    # Calculate and print the number of days the mission lasted
    difference = ctd_time_dt.max() - ctd_time_dt.min()
    num_days = difference.days
    print(f'The mission lasted {num_days} days.') 

    # Convert aa4831_time to the same format
    aa4831_time_dt = pd.to_datetime(ds.aa4831_time.values, unit='s', origin='unix')
    oxy_time_timestamps = aa4831_time_dt.astype(np.int64) // 10**9

    # Interpolate ctd_depth onto aa4831_time
    aa4831_depth = np.interp(aa4831_time_dt.astype(np.int64), ctd_time.astype(np.int64), ds.ctd_depth)

    # Time vs Depth Grid
    xn1, yn1 = int(num_days * 4), 200 # (The number of mission days multiplied by 4 dives per day (on average), 1000m / 5m per dive = 200 points)
    xmin1, xmax1 = ctd_time_timestamps.min(), ctd_time_timestamps.max()
    ymin1, ymax1 = 0, 1000
    xgrid1 = np.linspace(xmin1, xmax1, xn1)
    ygrid1 = np.linspace(ymin1, ymax1, yn1)
    Xgrid1, Ygrid1 = np.meshgrid(xgrid1, ygrid1)

    # Time vs Depth Grid (using the ctd_data_point dimension)
    xn2, yn2 = int(num_days * 4), 200 # (The number of mission days multiplied by 4 dives per day (on average), 1000m / 5m per dive = 200 points)
    xmin2, xmax2 = oxy_time_timestamps.min(), oxy_time_timestamps.max()
    ymin2, ymax2 = 0, 1000
    xgrid2 = np.linspace(xmin2, xmax2, xn2)
    ygrid2 = np.linspace(ymin2, ymax2, yn2)
    Xgrid2, Ygrid2 = np.meshgrid(xgrid2, ygrid2)


    # For loop to run through each variable in "sci_variables" and plot:
    #   1) Raw scatter plot
    #   2) Gridded scatter plot
    #   3) Gridded contour plot

    for var_name, var in sci_variables.items():
        print(f'Processing {var_name} data....')

        # Create a directory for the variable
        var_directory = figures_folder + f'\{var_name}'
        os.makedirs(var_directory, exist_ok=True)  # exist_ok=True prevents errors if the folder already exists

        # Choose specific cmap
        if var_name == 'temperature':
            cmap = cmocean.cm.thermal
        elif var_name == 'salinity':
            cmap = cmocean.cm.haline
        elif var_name == 'oxygen':
            cmap = cmocean.cm.oxy

        if var_name != 'oxygen':
            # Create an interpolated variables dataset
            interpolated_vars = {}
            interpolated_vars[f'{var_name}_interp'] = griddata((ctd_time_timestamps, ctd_depth.values), var, (Xgrid1, Ygrid1), method='linear')
           
            # Define x and y
            x = Xgrid1
            y = Ygrid1

            # Raw Scatter Plot
            plt.figure(figsize=(10, 5), dpi=300)
            plt.scatter(ctd_time, ctd_depth, c=var, cmap=cmap)
            plt.colorbar(label=f'{var.units}')
            plt.gca().invert_yaxis()
            plt.title(f'{time_coverage_start} - {time_coverage_end}')
            plt.xlabel('Time')
            plt.ylabel('Depth (m)')
            if var_name == 'temperature':
                clim = plt.clim(4,16)
            elif var_name == 'salinity':
                clim = plt.clim(32.25,34.50)
            plt.ylim(1000,0)
            plt.grid(alpha=0.5)
            plt.savefig(f'{var_directory}/raw_{var_name}_mission.png')
            plt.close()
            print(f'raw_{var_name}_mission.png created' )

        else:
            interpolated_vars[f'{var_name}_interp'] = griddata((oxy_time_timestamps, aa4831_depth), var, (Xgrid2, Ygrid2), method='linear')
            x = Xgrid2
            y = Ygrid2

            # Raw Scatter Plot
            plt.figure(figsize=(10, 5), dpi=300)
            plt.scatter(aa4831_time_dt, aa4831_depth, c=var, cmap=cmap)
            plt.colorbar(label=f'{var.units}')
            plt.gca().invert_yaxis()
            plt.title(f'{time_coverage_start} - {time_coverage_end}')
            plt.xlabel('Time')
            plt.ylabel('Depth (m)')
            plt.clim(0,300)
            plt.ylim(1000,0)
            plt.grid(alpha=0.5)
            plt.savefig(f'{var_directory}/raw_{var_name}_mission.png')
            plt.close()
            print(f'raw_{var_name}_mission.png created' )

        # Apply a rolling mean to the data [20m by 3 days]
        rolling_var = {}
        rolling_var[f'{var_name}'] = pd.DataFrame(interpolated_vars[f'{var_name}_interp'])
        rolling_var[f'{var_name}'] = rolling_var[f'{var_name}'].T
        rolling_var = rolling_var[f'{var_name}'].rolling(window=3, center=True, win_type='boxcar').mean() # window=3 is 3 days interpolation
        rolling_var = rolling_var.T
        rolling_var = rolling_var.rolling(window=4, center=True, win_type='boxcar').mean() # window=4 is 20m interpolation

        # Gridded Scatter Plot
        plt.figure(figsize=(10, 5), dpi=300)
        plt.scatter(x, y, c=rolling_var.values, cmap=cmap)
        plt.colorbar(label=f'{var.units}')
        plt.gca().invert_yaxis()
        plt.title(f'{time_coverage_start} - {time_coverage_end}')
        plt.xlabel('Time')
        plt.ylabel('Depth (m)')
        if var_name == 'temperature':
            clim = plt.clim(4,16)
        elif var_name == 'salinity':
            clim = plt.clim(32.25,34.50)
        elif var_name == 'oxygen':
            clim = plt.clim(0,300)
        plt.clim(clim)
        plt.ylim(1000,0)
        plt.grid(alpha=0.5)
        plt.savefig(f'{var_directory}/gridded_{var_name}_mission.png')
        plt.close()
        print(f'gridded_{var_name}_mission.png created' )

        # Gridded Contour Plot
        # Define levels per variable
        if var_name == 'temperature':
            levels = np.arange(2, 18, 2)
        elif var_name == 'salinity':
            levels = np.arange(31.5, 35, 0.5)
        elif var_name == 'oxygen':
            levels = np.arange(0,300,50)

        plt.figure(figsize=(10, 5), dpi=300)
        contour = plt.contourf(x, y, rolling_var, levels=levels, cmap=cmap)
        contour_lines = plt.contour(x, y, rolling_var.values, levels=levels, colors='black', linewidths=0.5)
        plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.1f')
        plt.colorbar(contour, label=f'{var.units}')
        plt.gca().invert_yaxis()
        plt.title(f'{time_coverage_start} - {time_coverage_end}')
        plt.xlabel('Time')
        plt.ylabel('Depth (m)')
        plt.ylim(1000,0)
        plt.grid(alpha=0.5)
        plt.savefig(f'{var_directory}/contour_{var_name}_mission.png')
        plt.close()
        print(f'contour_{var_name}_mission.png created' )

    print('Done!')

sg_timeseries(filepath, figures_folder)
