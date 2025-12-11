# Function to split seaglider dive data into ascent and descent phases
# Author: Jace Marquardt
# Last updated 2025-02-11

import numpy as np

def split_sg_profile(ds, threshold=0.07):
    """
    Splits seaglider dive data into ascent and descent phases.
    
    Parameters:
        ds (xr.Dataset): Dataset containing 'ctd_time' and 'ctd_depth' variables.
        
    Returns:
        dive (xr.Dataset): Dataset containing descent data.
        climb (xr.Dataset): Dataset containing ascent data.
    """

    ds = ds.sortby('ctd_time')

    # Convert ctd_time to numerical format (Unix epoch time in seconds)
    ds = ds.assign_coords(
        ctd_time=(ds["ctd_time"] - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    )

    depth_diff = ds['ctd_depth'].differentiate('ctd_time')
    
    # Identify ascent and descent using the threshold
    dive = ds.where(depth_diff > threshold, drop=True)
    climb = ds.where(depth_diff < -threshold, drop=True)
    
    return dive, climb