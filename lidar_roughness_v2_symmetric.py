# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:17:14 2018
This script will take in a lidar slice from cloud compare, converted into a ascii format
of above ground filtered vege only, and do the manners/ straatsma et al 2008

In this version I will use the symmetric equation of a 3d line

Note because we are reorienting the scanner to be the origin, the dx, dy , and dz numbers are the corresponding
A, B, and C vectors in the


@author: jgarber
"""

import numpy as np
import matplotlib.pyplot as plt

#note have to find these coords in cyclone, need to make sure to write them all down!
st_e, st_n, st_z =   (428256.438, 7015201.067, 114.47)

#scan is high resolution
spacing = 0.05
l_angle = np.arctan(spacing/100)

#!!! Box dimensions
b_dim = 1

lidar = np.loadtxt('off_ground_points.txt', delimiter = ',', skiprows = 1)

#shift things to be relatvie to the station

dx = lidar[:,0] - st_e
dy = lidar[:,1] - st_n
dz = lidar[:,2] - st_z

#convert to spherical coordinates
def Sphere_coord(dx,dy,dz):
    h_a = np.arctan(dy/dx)  #horizontal angle
    hd = np.sqrt(dx**2+dy**2)   #horizontal distance
    z_a = np.arctan(dz/hd)      #vertical angle
    R = np.sqrt(hd**2+dz**2)    #vstraight line distance
    return h_a, hd, z_a, R

h_a, hd, z_a, R = Sphere_coord(dx,dy,dz)

Par = np.vstack((dx,dy,dz,z_a,h_a))


#make a grid of hangles and z angles from all shots fired

r_z_min = np.min(z_a)
r_h_min =np.min(h_a)
r_z_max = np.max(z_a)
r_h_max = np.max(h_a)

#horizontal and vertical angular spacings of all of the lasers


#symetric unit vectors
def sim_from_angle(ha_l,za_l):
    l_A = np.cos(ha_l)
    l_B = np.sin(ha_l)
    l_C = np.sin(za_l)
    h_a = len(l_A)
    h_c = len(l_C)
    l_A = np.broadcast_to(l_A[np.newaxis,:],(h_c,h_a)).flatten()
    l_B = np.broadcast_to(l_B[np.newaxis,:],(h_c,h_a)).flatten()
    l_C = np.broadcast_to(l_C[:,np.newaxis],(h_c,h_a)).flatten()

    return l_A,l_B,l_C



def spotlight(X,Y,Z,b_dim= b_dim, l_angle=l_angle):
    """
    This function creates a spotlight of laser shots on a cube usign the cube's
    center point. therefore I don't have to subset everything'
    """
    h_a, hd, z_a, R = Sphere_coord(X+b_dim/2,Y+b_dim/2,
                                   np.array([Z+b_dim/2,Z+b_dim*2]))
    theta = np.abs(z_a[0]-z_a[1])
    z_a = z_a[0]
    #theta = np.arcsin(np.abs(b_dim*3/R))
    r_h_min = np.floor((h_a-theta)/l_angle)*l_angle
    r_h_max = np.ceil((h_a+theta)/l_angle)*l_angle
    r_z_min = np.floor((z_a-theta)/l_angle)*l_angle
    r_z_max = np.ceil((z_a+theta)/l_angle)*l_angle
    ha_l = np.arange(r_h_min,r_h_max,l_angle)
    za_l = np.arange(r_z_min,r_z_max,l_angle)
    A,B,C = sim_from_angle(ha_l,za_l)
    return A,B,C, (r_z_min, r_z_max), (r_h_min, r_h_max)

def overlap(Xr,Yr):
    '''
    This is where we a 2X number of lasers array
    2 tuples indicating the boundaries of the symmetric equation
    and find out if they overlap
    '''
    out = np.abs(Yr[1]) <0 # note should create a false array
    #goign after at by finding out which range is bigger first
    Xo = np.abs(Xr[1]-Xr[0]) > np.abs(Yr[1]-Yr[0])

    Xo_in_r = ((Yr[0] < Xr[1]) & (Yr[0]>=Xr[0])) | ((Yr[1] < Xr[1]) & (Yr[1]>=Xr[0]))

    out[Xo & Xo_in_r] = True


    Yo_in_r = ((Xr[0] < Yr[1]) & (Xr[0]>=Yr[0])) | ((Xr[1] < Yr[1]) & (Xr[1]>=Yr[0]))

    out[(~Xo) & Yo_in_r] = True

    out[(Xr[0] ==Yr[0])] = True

    return out

def pass_through(Xb,Yb,Zb,A,B,C, bdim=b_dim):
    """
    This is the function where we take the symmetric equations, and find
    out if we pass through them
    """
    X = np.array([Xb, Xb+bdim])
    Y = np.array([Yb, Yb+bdim])
    Z = np.array([Zb, Zb+bdim])
    Xr = X[:,np.newaxis]/A[np.newaxis,:]
    Yr = Y[:,np.newaxis]/B[np.newaxis,:]
    Zr = Z[:,np.newaxis]/C[np.newaxis,:]
    #when one of them is along an axis, y = y0
    Xr[:,(A == 0)] = 0
    Yr[:,(B==0)] = 0
    Zr[:,(C==0)] = 0
    #print(Xr.shape, Yr.shape,Zr.shape)
    X_t_Y = overlap(Xr,Yr)
    X_t_Z = overlap(Xr,Zr)
    Y_t_Z = overlap(Yr,Zr)

    intersects = (X_t_Y & X_t_Z) & Y_t_Z
    return intersects

def not_reach_box(Xb,Yb,Zb,Xp,Yp,Zp,bdim=b_dim):
    X = np.array([Xb, Xb+bdim])
    Y = np.array([Yb, Yb+bdim])
    Z = np.array([Zb, Zb+bdim])
    lowx = Xp< np.min(X)
    lowy = Yp< np.min(Y)
    lowz = Zp< np.min(Z)
    out = lowx |lowy |lowz
    return out



def analyse_data(Xb,Yb,Zb,P_ar, b_dim = b_dim, l_angle = l_angle):
    """
    This is where I run the whole analysis, and get back
    T - total number of pulses that should pass through
    P - the number that actually pass through
    B - number of points in voxel
    PAR has 0th dimensions corresponding to different point features
    0 - X, 1-Y,2-Z, 3-Zangle, 4-h_angle
    """


    #this is the part where they churn through the potential laser points
    A,B,C, z_a_r, h_a_r = spotlight(Xb,Yb,Zb,b_dim= b_dim, l_angle=l_angle)
    T = np.sum(pass_through(Xb,Yb,Zb,A,B,C, bdim=b_dim))

    #next is to filter out the actual laser points vectoring through the spotlight
    # !!! Note as of 29/11/18 cannot get this spotlight filtering to work
    #pfilt = ((P_ar[3]<=z_a_r[1]) &(P_ar[3]>=z_a_r[0])) & ((P_ar[4]<=h_a_r[1]) &(P_ar[4]<=z_a_r[0]))
    # P_ar = P_ar[:,pfilt]

    Xp = P_ar[0]
    Yp= P_ar[1]
    Zp= P_ar[2]

    B = np.sum(((Xp>= Xb) & (Xp< (Xb+b_dim))) & ((Yp>= Yb) & (Yp< (Yb+b_dim))) & ((Zp>= Zb) & (Zp< (Zb+b_dim))))

    if B == 0:
        P = 0
        return T, P, B




    else:
        intersects = pass_through(Xb,Yb,Zb,Xp,Yp,Zp, bdim=b_dim)


        shorts =  not_reach_box(Xb,Yb,Zb,Xp[intersects],Yp[intersects],Zp[intersects],bdim=b_dim)

        P = np.sum(intersects) - np.sum(shorts) # the sume of how many things make it vs how many of our points went through minus the ones taht didnt make it there

        #print('Points passing through = ',P)
        #print('Points in  =',B)
        #B = np.sum(((Xp>= Xb) & (Xp< Xb+b_dim)) & ((Yp>= Yb) & (Yp< Yb+b_dim)) & ((Zp>= Zb) & (Zp< Zb+b_dim)))

        return T, P, B



#ok time to make the grid
dx_range = (np.floor(np.min(dx)/b_dim)*b_dim,np.ceil(np.max(dx)/b_dim)*b_dim)
dy_range = (np.floor(np.min(dy)/b_dim)*b_dim,np.ceil(np.max(dy)/b_dim)*b_dim)
dz_range = (np.floor(np.min(dz)/b_dim)*b_dim,np.ceil(np.max(dz)/b_dim)*b_dim)

X_g = np.arange(dx_range[0],dx_range[1],b_dim)
Y_g = np.arange(dy_range[0],dy_range[1],b_dim)
Z_g = np.arange(dz_range[0],dz_range[1],b_dim)

shell = np.zeros((len(X_g),len(Y_g),len(Z_g)))

X_g = np.broadcast_to(X_g[:,np.newaxis],shell.shape[0:2])
X_g = np.broadcast_to(X_g[:,:,np.newaxis],shell.shape)

Y_g = np.broadcast_to(Y_g[np.newaxis,:],shell.shape[0:2])
Y_g = np.broadcast_to(Y_g[:,:,np.newaxis],shell.shape)

Z_g = np.broadcast_to(Z_g[np.newaxis,:],shell.shape[1:])
Z_g = np.broadcast_to(Z_g[np.newaxis,:,:],shell.shape)



#output arrays
T_ar = shell*0
P_ar = shell*0
B_ar = shell*0

"""
!!! NOte end of resbaz restrain to Warnambool

Something is wrong with the voxeling, when I make a voxel with a corner
below a point, it seems to work , but when I make it loop through a premade grid
of Voxels, it doesn't seem to pick anything up Will have to crack it on the way back!!'
"""


it = np.nditer(shell,flags = ['multi_index'])
import time
begin = time.time()
while not it.finished:
     ijk = it.multi_index
     Xb = X_g[ijk]
     Yb = Y_g[ijk]
     Zb = Z_g[ijk]
     #print(Zb,Yb,Zb)
     if ((Xb==0.) & (Yb==0.)) & (Zb==0.):
         T_ar[ijk], P_ar[ijk], B_ar[ijk] = 0,0,0
     else:
         T_ar[ijk], P_ar[ijk], B_ar[ijk] = analyse_data(Xb,Yb,Zb,Par, b_dim = b_dim, l_angle = l_angle)

     it.iternext()

end = time.time()

elapsed = end - begin


P_arf = P_ar.flatten()
B_arf = B_ar.flatten()
T_arf = T_ar.flatten()

Percblock = (B_arf/P_arf)

overpoint = np.sum(Percblock>1)

plt.scatter((B_arf/P_arf), (P_arf/T_arf))

#plt.hist((B_arf/P_arf,

import pickle

out_Dict = {"P":P_ar,'B':B_ar,'T':T_ar,'Ranges':[dx_range,dy_range,dz_range]}

file = open('emu_pickle','wb')

pickle.dump(out_Dict,file)




### Test the analyse function
'''
P_ar = Par
Xb = dx[50]-0.1
Yb = dy[50]-0.1
Zb = dz[50]-0.1
A,B,C, z_a_r, h_a_r = spotlight(Xb,Yb,Zb,b_dim= b_dim, l_angle=l_angle)
T = np.sum(pass_through(Xb,Yb,Zb,A,B,C, bdim=b_dim))
 #next is to filter out the actual laser points
#pfilt = ((P_ar[3]<=z_a_r[1]) & (P_ar[3]>=z_a_r[0])) & ((P_ar[4]<=h_a_r[1]) &(P_ar[4]<=z_a_r[0]))
#P_ar = P_ar[:,pfilt]


Xp = P_ar[0]
Yp= P_ar[1]
Zp= P_ar[1]

B = np.sum(((Xp>= Xb) & (Xp< (Xb+b_dim))) & ((Yp>= Yb) & (Yp< (Yb+b_dim))) & ((Zp>= Zb) & (Zp< (Zb+b_dim))))

intersects = pass_through(Xb,Yb,Zb,Xp,Yp,Zp, bdim=b_dim)


shorts =  not_reach_box(Xb,Yb,Zb,Xp[intersects],Yp[intersects],Zp[intersects],bdim=b_dim)

P = np.sum(intersects) - np.sum(shorts) # the sume of how many things make it vs how many of our points went through minus the ones taht didnt make it there
'''