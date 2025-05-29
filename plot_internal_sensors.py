# Function to plot humidity, temperature, and internal pressure from log files
# Author: Jace Marquardt
# Last Updated: 2025-05-28

import os
import matplotlib.pyplot as plt
from parse_log_file import parse_log_file

def plot_internal_sensors(filepath):
    """
    Plots humidity, temperature, and internal pressure from log files in the specified directory.
    
    Parameters:
        filepath (str): Path to the directory containing log files.
        
    Returns:
        None: Saves a plot of internal sensors.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The directory {filepath} does not exist.")
    
    dive_numbers = []
    humidity = []
    temperature = []
    internal_pressure = []

    for file in os.listdir(filepath):
        if file.endswith('.log'):
            file_path = os.path.join(filepath, file)  # Use a new variable
            log_parameters = parse_log_file(file_path)

            dive = log_parameters['DIVE']
            dive_numbers.append(dive)

            humid = log_parameters.get('HUMID', 0)
            temp = log_parameters.get('TEMP', 0)
            int_pres = log_parameters.get('INTERNAL_PRESSURE', 0)
            
            humidity.append(humid)
            temperature.append(temp)
            internal_pressure.append(int_pres)

            print(f"Parsing: {file}")

    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
    humid_plot = ax.plot(dive_numbers, humidity, linestyle='-', color='b', label='Humidity (%)')
    ax2 = ax.twinx()  # Create a second y-axis for temperature and internal pressure
    temp_plot = ax2.plot(dive_numbers, temperature, linestyle='-', color='r', label='Temperature (°C)')
    ax3 = ax.twinx()  # Create a third y-axis for internal pressure
    ax3.spines['right'].set_position(('outward', 60))  # Offset the third y-axis
    int_pres_plot = ax3.plot(dive_numbers, internal_pressure, linestyle='-', color='purple', label='Internal Pressure (psia)')
    ax.invert_yaxis()  # Invert y-axis for depth
    ax.set_xlabel('Dive Number')
    ax.set_ylabel('Humidity (%)')
    ax2.set_ylabel('Temperature (°C)')
    ax3.set_ylabel('Internal Pressure (psia)')
    ax.set_title('Internal Sensors')
    ax.legend(handles=[humid_plot[0], temp_plot[0], int_pres_plot[0]], loc='upper right')
    ax.grid(axis='both', which='both', linestyle='--', linewidth=0.7)
    plt.xlim(min(dive_numbers), max(dive_numbers)+10)
    plt.tight_layout()
    plt.savefig(os.path.join(filepath, 'internal_sensors_plot.png'))
    print(f"Saving figure: {os.path.join(filepath, 'internal_sensors_plot.png')}.")

# Example usage
filepath = r'C:\Users\marqjace\TH_line\deployments\mar_2025\transect4\logfiles'
plot_internal_sensors(filepath)