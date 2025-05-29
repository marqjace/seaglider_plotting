# Function to plot bottom depths from log files
# Author: Jace Marquardt
# Last Updated: 2025-05-28

import os
import numpy as np
import matplotlib.pyplot as plt
from parse_log_file import parse_log_file

def plot_bottom_depths(filepath):
    """
    Plots bottom depths from log files in the specified directory.
    
    Parameters:
        filepath (str): Path to the directory containing log files.
        
    Returns:
        None: Saves a plot of bottom depths.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The directory {filepath} does not exist.")
    
    bottom_depths = []
    dive_numbers = []

    for file in os.listdir(filepath):
        if file.endswith('.log'):
            file_path = os.path.join(filepath, file)
            log_parameters = parse_log_file(file_path)

            dive = log_parameters['DIVE']
            dive_numbers.append(dive)

            bottom_depth = log_parameters.get('ALTIM_BOTTOM_PING', [0, 0])
            bottom_depth = bottom_depth[0] + bottom_depth[1]
            
            if 'ALTIM_BOTTOM_PING' not in log_parameters:
                # Handle missing value: use a default, raise a warning, etc.
                bottom_depth = np.nan  # or some sensible default

            bottom_depths.append(bottom_depth)
            print(f"Parsing: {file}")

    # Assign colors: red if depth is 1000, blue otherwise
    colors = ['r' if depth == 1000 else 'b' for depth in bottom_depths]

    plt.figure(figsize=(12, 8), dpi=300)
    plt.scatter(dive_numbers, bottom_depths, marker='o', linestyle='-', color=colors)
    plt.gca().invert_yaxis()  # Invert y-axis for depth
    plt.ylim(1000, 0)  # Set y-axis limits to 0-1000m
    plt.grid(which='both', linestyle='--', linewidth=0.7)
    plt.xlabel('Dive Number')
    plt.ylabel('Bottom Depth (m)')
    plt.title('Bottom Depths from Log Files')
    plt.tight_layout()
    plt.savefig(os.path.join(filepath, 'bottom_depths_plot.png'))
    print(f"Saving figure: {os.path.join(filepath, 'bottom_depths_plot.png')}.")

# Example usage
filepath = r'C:\Users\marqjace\TH_line\deployments\mar_2025\transect4\logfiles'
plot_bottom_depths(filepath)