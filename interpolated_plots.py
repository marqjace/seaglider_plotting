# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Imports

import matplotlib.pyplot as plt
import numpy as np
import scipy
import xarray as xr
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D


#%%
# Open Dataset

glider_data = xr.open_dataset("C:/Users/marqjace/OneDrive - Oregon State University/Desktop/Python/p6850630.nc")


# Set Variables

lat = glider_data['latitude'].values
lon = glider_data['longitude'].values
time = glider_data['time'].values
temp = glider_data['temperature'].values
depth = glider_data['depth'].values
sal = glider_data['salinity'].values
dens = glider_data['density'].values


# Interpolation With Depth

z_new = np.arange(0, 1000, 1)
f = scipy.interpolate.interp1d(depth, temp, kind='linear', bounds_error=False)
temp_new = f(z_new)

f = scipy.interpolate.interp1d(depth, sal, kind='linear', bounds_error=False)
sal_new = f(z_new)

f = scipy.interpolate.interp1d(depth, dens, kind='linear', bounds_error=False)
dens_new = f(z_new)

f = scipy.interpolate.interp1d(depth, lon, kind='linear', bounds_error=False)
lon_new = f(z_new)

f = scipy.interpolate.interp1d(lon, temp, kind='linear', bounds_error=False)
temp_new_2 = f(lon_new)


#%%

# Set Up Figure (Temperature Profile

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), dpi=300)
plt.suptitle('Temperature Profile')


# Interpolated Plot

ax1.plot(temp_new, z_new)
ax1.invert_yaxis()
ax1.set_ylabel('Depth (m)')
ax1.set_xlabel('Temperature (deg C)')
ax1.set_title('Interpolated')


# Raw Plot

ax2.plot(temp, depth, c='r')
ax2.invert_yaxis()
ax2.set_ylabel('Depth (m)')
ax2.set_xlabel('Temperature (deg C)')
ax2.set_title('Raw')


# Save Figure
plt.savefig('C:/Users/marqjace/OneDrive - Oregon State University/Desktop/Python/Figure3.jpg')

#%%

# Set Up Figure2 (Salinity Profile)

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 8), dpi=300)
plt.suptitle('Salinity Profile')


# Interpolated Plot

ax3.plot(sal_new, z_new)
ax3.invert_yaxis()
ax3.set_ylabel('Depth (m)')
ax3.set_xlabel('Salinity (PSU)')
ax3.set_title('Interpolated')


# Raw Plot

ax4.plot(sal, depth, c='r')
ax4.invert_yaxis()
ax4.set_ylabel('Depth (m)')
ax4.set_xlabel('Salinity (PSU)')
ax4.set_title('Raw')


# Save Figure
plt.savefig('C:/Users/marqjace/OneDrive - Oregon State University/Desktop/Python/Figure4.jpg')


#%%

# Set Up Figure3 (Density Profile)

fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 8), dpi=300)
plt.suptitle('Density Profile')


# Interpolated Plot

ax5.plot(dens_new, z_new)
ax5.invert_yaxis()
ax5.set_ylabel('Depth (m)')
ax5.set_xlabel(r'Density (kg/$m^3$)')
ax5.set_title('Interpolated')


# Raw Plot

ax6.plot(dens, depth, c='r')
ax6.invert_yaxis()
ax6.set_ylabel('Depth (m)')
ax6.set_xlabel(r'Density (kg/$m^3$)')
ax6.set_title('Raw')


# Save Figure
plt.savefig('C:/Users/marqjace/OneDrive - Oregon State University/Desktop/Python/Figure5.jpg')


#%%

# Set Up Figure 4 (Colormap Example)

fig4, ax7 = plt.subplots(1, 1, dpi=300)

color_example = ax7.scatter(lon_new, z_new, c=temp_new, cmap='coolwarm')
ax7.invert_yaxis()
ax7.set_ylabel('Depth (m)')
ax7.set_xlabel('Longitude')
ax7.set_title('Interpolated Profile')
fig.colorbar(color_example, label=r'Temperature ($\degree$C)')
plt.savefig('C:/Users/marqjace/OneDrive - Oregon State University/Desktop/Python/Figure6.jpg')


#%%

# Set Up Figure 5 (Colormap Example 2)

fig5, ax8 = plt.subplots(1, 1, dpi=300)

color_example2 = ax8.scatter(lon, depth, c=temp, cmap='coolwarm')
ax8.invert_yaxis()
ax8.set_ylabel('Depth (m)')
ax8.set_xlabel('Longitude')
ax8.set_title('Raw Profile')
fig.colorbar(color_example2, label=r'Temperature ($\degree$C)')
plt.savefig('C:/Users/marqjace/OneDrive - Oregon State University/Desktop/Python/Figure7.jpg')


#%%

# Set Up Figure 6 (Surface Plot)

Lon, Depth = np.meshgrid(lon, depth)
# Temp, Depth2 = np.meshgrid(temp, depth)
#
# fig6 = plt.figure()
# ax9 = fig.add_subplot(111, projection='3d')
# ax9.plot_surface(Lon, Depth, Temp, cmap='coolwarm')
# plt.savefig('C:/Users/marqjace/OneDrive - Oregon State University/Desktop/Python/Figure8.jpg')

ax9 = fig.add_subplot(111, projection='3d')
ax9.plot_surface(lon, lat, Depth)

# ax9 = plt.gca()
# ax9.hold()
# ax9.scatter(lon, lat, temp)
plt.savefig('C:/Users/marqjace/OneDrive - Oregon State University/Desktop/Python/Figure8.jpg')


