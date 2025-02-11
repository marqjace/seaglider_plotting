# Function to split seaglider dive data into ascent and descent phases
# Author: Jace Marquardt
# Last updated 2025-02-11

import numpy as np

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