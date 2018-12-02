# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:17:14 2018
This script will take in a lidar slice from cloud compare, converted into a ascii format
of above ground filtered vege only, and do the manners/ straatsma et al 2008
 voxelization of everything, co
@author: jgarber
"""

import numpy as np

#note have to find these coords in cyclone, need to make sure to write them all down!
st_e, st_n, st_z =   (428256.438, 7015201.067, 114.47)


lidar = np.loadtxt('off_ground_points.txt', delimiter = ',', skiprows = 1)

#shift things to be relatvie to the station

dx = st_e - lidar[:,0]
dy = st_n - lidar[:,1]
dz = st_z - lidar[:,2]

#convert to spherical coordinates

h_a = np.arctan(dx/dy)  #horizontal angle
hd = np.sqrt(dx**2+dy**2)   #horizontal distance
z_a = np.arctan(dz/hd)      #vertical angle
R = np.sqrt(hd**2+dz**2)    #vstraight line distance

angle_d = 2
angle_a = angle_d*3.1459/180

r_z = 0
r_h = 0
rz_max = np.max(z_a)
rh_max = np.max(h_a)

n_z = int(rz_max/angle_d)+1
n_h = int(rh_max/angle_d)+1


R_bins = np.arange( 0,max(R)+0.2,0.2)



#!!!!need to creat an empty natrix here with np.zeros
out = np.zeros(len(R_bins),n_z*n_h)
i = 0
while (r_z < rz_max) and (i < n_h*n_z):
    while r_h < rh_max:
        #willsubset the R with the angles and then bin it
        boo_ar = ((h_a >= r_h) & (h_a < (r_h+angle_a))) & ((z_a >= r_z) & (z_a < (r_z+angle_a)))
        out[:,i] = np.histogram(R[boo_ar],bins=R_bins)

        r_h +=angle_a
        i+=1
    r_z +=angle_a
    r_h = 0


