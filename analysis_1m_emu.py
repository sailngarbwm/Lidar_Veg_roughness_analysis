# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 16:37:47 2018

@author: jgarber
"""

import pickle

import numpy as np
import matplotlib.pyplot as plt

import xarray as xr


file = open('emu_pickle','rb')
emu_dict = pickle.load(file)

P = emu_dict['P']
B = emu_dict['B']
T = emu_dict['T']
dx_range, dy_range,dz_range = emu_dict['Ranges']

p_return = B[:]*0
p_return[(B>0) |(P>0)] = B[(B>0) |(P>0)]/(P[(B>0) |(P>0)]+B[(B>0) |(P>0)])



plt.hist(p_return.flatten())

B_sum = np.sum(B,axis=2)

plt.imshow(B_sum)

T_sum = np.sum(T,axis=2)

plt.imshow(T[:,:,6])

b_dim = 1
X_g = np.arange(dx_range[0],dx_range[1],b_dim)
Y_g = np.arange(dy_range[0],dy_range[1],b_dim)
Z_g = np.arange(dz_range[0],dz_range[1],b_dim)

ds = xr.Dataset({'B': (['x', 'y', 'z'],  B),
                 'P': (['x', 'y', 'z'], P),
                 'T': (['x', 'y', 'z'], T)},
                 coords={'Easting': (['x'], X_g),
                         'Northing': ([ 'y'], Y_g),
                         'Elev':(['z'],Z_g)})

ds.to_netcdf('emu_rough_1m.nc')

import geoviews as gv

gvds = gv.Dataset(ds,kdims = ['Easting','Northing',])