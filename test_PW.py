import numpy as np
from math import pi as pi
from math import cos as cos
from math import sin as sin
import math
from pylab import *
import matplotlib.pyplot as plt

f = 2000        # frequency in Hertz
c = 343         # speed of sound

#--------------- Set the array geometry -----------------

arrtype = 'linear'
#arrtype = 'circular'
num_mics = 5    # number of microphones
d = 1*0.1         # distance between microphones

if arrtype == 'linear':
    tmp = np.arange(0,num_mics*d,d)
    tmp = tmp[0:num_mics]
    apperture_len = d*(num_mics-1)
    xcoord = tmp-apperture_len/2
    ycoord = np.zeros(shape = (num_mics))
    arrcenter = [0,0];
    mic_pos = [xcoord, ycoord]
    mic_pos = np.matrix(mic_pos)
else:
    tmp = np.arange(0,2*pi,2*pi/num_mics)
    mic_pos = np.matrix([cos(tmp), sin(tmp)])
    diameter = d*(num_mics-1) # similar apperture_len as a linear with same number of elements
    mic_pos = 0.5*diameter*mic_pos



#---------------- Set the source location ---------------

source_dist = 1.5 # source distance from array centre
source_angle = 0*pi/180  # source angle w.r.t. array centre (x-axis 0 reference)


source_doa = np.matrix([cos(source_angle), sin(source_angle)])
source_pos = source_dist*source_doa

# Plot the selected room geometry
fig,ax1 = plt.subplots(1,1,figsize = (7,7))
#ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax1.plot(mic_pos[0,:],mic_pos[1,:],'o', color = 'r', markerfacecolor = 'y')
ax1.plot(source_pos[0,0],source_pos[0,1], 's',color = 'b', markersize=10, markerfacecolor = 'b')
ax1.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
ax1.set_xlim(-2,2)
ax1.set_ylim(-2,2)
ax1.set_aspect('equal', 'box')


#-------- CREATE THE STEERING TABLE FOR A GIVEN ARRAY GEOMETRY----------

angle_resolution = 2 # What is the angle resolution in degrees
scan_angles = np.arange(-180,180,angle_resolution)*pi/180

# Unit vectors describing different directions of propagation
scan_vectors = np.matrix([np.cos(scan_angles),np.sin(scan_angles)])

# Get the dot-products (see the formula  for plane waves for clarification)
# size: number_of_scan_angles x number_of_microphones
dotproducts = np.transpose(scan_vectors)*mic_pos

# Now, set the frequency axis to be scanned
freq_Hz = np.arange(0,8000,100) # modify it later to fit the STFT
freq_Hz = freq_Hz[1:freq_Hz.shape[0]]
lambd = np.divide(c,freq_Hz)
kappa_scan = 2*pi/lambd

# Now, we need to create all the exponentials for all wavenumbers, angles, and sensor locations (3D tensor)
tmp = np.zeros((dotproducts.shape[0],dotproducts.shape[1],1))
tmp[:,:,0] = dotproducts
dotproducts = tmp

all_phases = kappa_scan*dotproducts

STEERING_TABLE = np.exp(-1j*all_phases)

#-------- CREATE A PLANE WAVE IN THE FREQUENCY DOMAIN------------

# set the wave FREQUENCY
fbin = 20
f = freq_Hz[fbin-1]


lambd = c/f
kappa = 2*pi/lambd
# The minus sign is required because the doa is negative of the the direction of propagation
# In the plane wave formula usually the direction of propagation is used
wavevector= kappa*source_doa
randphase = np.random.rand(1)*2*pi-pi

# The complex pressure at the microphone due to this plane wave
Xf = np.exp(1j*(wavevector*mic_pos + randphase))

# --------CHECK THE BEAMPATTERN AT THE SELECTED FREQUENCY------------

# shape: scan_angles x num_mics
Beamformers = np.matrix(STEERING_TABLE[:,:,fbin-1])
BF_response = Xf*np.transpose(Beamformers)/num_mics

# the output power (narrowband steered response power)
SRP = np.transpose(np.absolute(BF_response))

# fig, ax = plt.subplots(1,1,figsize = (10,10))
# ax.plot(scan_angles*180/pi,SRP)
# ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
# ax.plot(source_angle*180/pi,1, 'o')
# ax.plot(-source_angle*180/pi,1, 'o')


# ------ POLAR PLOT OF THE RESULT -----

SRP_db = 10*np.log(SRP)
flags = SRP_db < - 16
SRP_db[flags] = -16


fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111, projection='polar')

ax.plot(scan_angles, SRP_db)
ax.set_rlabel_position(source_angle*180/pi + 180)

#ax.set_rticks([0.4,0.6,0.8,1])  #Radial ticks

# scan_wavenums = kappa*scan_vec
# tmp = np.transpose(scan_wavenums)*mic_pos
# BF_vector = np.exp(1j*tmp)
#
# # Now multiply the BF vector with the frequency domain pressure signal X_f
# BF_response = X_f*BF_vector.getH()/num_mics
# SRP = np.transpose(np.absolute(BF_response))
#
#
# # -------- Polar plot of the response ------------
# fig, ax = plt.subplots(1,1,figsize = (10,10))
# ax.plot(scan_angles*180/pi,SRP)
#
# ax.grid(color = 'r',linestyle = '-')
