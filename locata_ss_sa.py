# Based on the task1 from the locata challenge: one static source, one static array

import numpy as np
from math import pi as pi
from math import cos as cos
from math import sin as sin
import math
from pylab import *
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
from module_stft import *
import matplotlib.animation as animation

# Load the static array geometry
my_data = genfromtxt('array_geometries/locata/arr_benchmark2_task2_recording2.dat', delimiter=',')
mic_pos = my_data[0:2,:] # this removes the z-coordinate, the current estimators do not consider elevation angle

# Load the multichannel audio file in numpy array
Fs, inputSig = wf.read('test_signals/locata/sig_benchmark2_task2_recording2.wav')
numchans = inputSig.shape[1]

#figure()
#plot(inputSig)

# Shorten the signal for quicker testing
#inputSig = inputSig[30000:100000,:]

# Set the STFT parameters
winsize_samples = 1024
nfft_samples = 1024
overlap_percentage = 50
wintype = 'cosine'

# ----THE STEERING TABLE FOR THIS ARRAY GEOMETRY AND FILTERBANK SETTINGS

# The funcion is in the stft module
c = 343
angle_resolution = 2
STEERING_TABLE, scan_angles = generate_steer_table(angle_resolution,mic_pos,nfft_samples,Fs)

# -----------------GET STFT OF THE SPEECH SIGNAL-------------
# The function is in the stft module

numFrames = math.ceil(inputSig.shape[0]/(winsize_samples-overlap_percentage*winsize_samples/100)) - 100/overlap_percentage
spectrum = 1j*np.zeros((int(nfft_samples/2+1),int(numFrames),numchans))
for idx in range(0,numchans):
    spectrum[:,:,idx] = stft(inputSig[:,idx],winsize_samples,nfft_samples,overlap_percentage,wintype)

## Plot the spectrum
#hopsize = int((100-overlap_percentage)*winsize_samples/100)
#taxis_spec = np.arange(0,spectrum.shape[1],1)*hopsize/Fs
#fig, axes = plt.subplots(2,1,figsize = (14,10))
#axes[0].pcolormesh(taxis_spec,freq_Hz, 10*np.log10(np.abs(spectrum[:,:,0])), cmap = 'jet')

#------------- INSTANTANEOUS PHASE DIFFERENCES OF THE OBSERVED SPECTRUM
#phases = np.angle(spectrum)
#phases.shape

# Create plane waves with this phases
#spectrum_hat = np.exp(1j*phases)
#spectrum_hat = np.expand_dims(spectrum_hat, axis=3)  # add one more dimension
#spectrum_t = spectrum_hat.transpose(0,2,1,3) # transpose to match some dimensions

#------------ CROSS POWER SPECTRAL DENSITY OF THE OBSERVED SPECTRUM

avg_const = 0.4 # exponential averaging for the PSD
phases_cpsd = zeros(shape = spectrum.shape)
phi_mvdr = zeros(shape =  (spectrum.shape[0],spectrum.shape[1], scan_angles.shape[0]))


for fidx in range(0,spectrum.shape[0]): # loop over frequencies

    # Get the steering vectors for this frequency
    dd = STEERING_TABLE[:,:,fidx]

    #Ryy = np.zeros(shape = (spectrum.shape[2],spectrum.shape[2])) # initialize the PSD matrix
    Ryy = np.identity(numchans)*0.0000001

    for idx in range(0,spectrum.shape[1]): # loop over frames

        tmpsig = np.expand_dims(spectrum[fidx,idx,:],axis = 1)
        tmpsig_h = np.matrix.getH(tmpsig)
        yyh = matmul(tmpsig,tmpsig_h)
        Ryy = avg_const*Ryy + (1-avg_const)*yyh

        # Now get the angle of the cross-PSD vector (e.g., first column)
        tmp_phase = np.angle(Ryy[:,0])
        phases_cpsd[fidx,idx,:] = tmp_phase

        # What follows is for the MVDR angular spectrum. Comment until the end of loop if not used
        # invRyy = np.linalg.inv(Ryy)
        # totalpower = np.real(np.matrix.trace(Ryy))/numchans
        #
        # # loop over angles to get the MVDR spectrum
        # for aidx in range(0,scan_angles.shape[0]):
        #
        #     dd_aidx = np.expand_dims(dd[aidx,:],axis = 1)
        #     quad = dot(dot(np.matrix.transpose(dd_aidx),invRyy),np.conj(dd_aidx))
        #     quadreal = np.real(quad)
        #     quadform = float(quadreal)
        #
        #     # the mvdr angular spectrum
        #     phi_mvdr[fidx,idx,aidx] = 1/(quadform*totalpower-1)


#----------- CREATE PLANE WAVES USING THE CPSD (REQURED FOR THE SRP)
spectrum_hat = np.exp(1j*phases_cpsd)
spectrum_hat= np.expand_dims(spectrum_hat, axis=3)  # add one more dimension
spectrum_hat_t = spectrum_hat.transpose(0,2,1,3) # transpose to match some dimensions


#---------- MATCH THE DIMENSIONS OF THE STEERING TABLE TO THE SPECTRA -------
steering_t = np.expand_dims(STEERING_TABLE, axis=3)
steering_t = steering_t.transpose(2,1,3,0)


#-----------------CREATING THE DIFFERENT ANGULAR SPECTRA-----------

# Standard delay-and-sum steering (SRP)
SRP = np.square(np.absolute(np.sum(steering_t*spectrum_hat_t,axis = 1)/numchans))

# Apply a nonlinearity to enhance peaks [Loesch2010, in IVA/LCA]
SRP_nonlin = 1-np.tanh(2*np.sqrt(1-SRP))

# The MVDR spectrum
MVDR = phi_mvdr

# -------- NARROWBAND DOA: search maximum per frequency --------
# This is subject to aliasing

DOA_nb_SRP = 180*scan_angles[SRP.argmax(axis = 2)]/pi
# DOA_nb_MVDR = 180*scan_angles[MVDR.argmax(axis = 2)]/pi

# Plot the narrowband estimates
hopsize = int((100-overlap_percentage)*winsize_samples/100)
taxis_spec = np.arange(0,spectrum.shape[1],1)*hopsize/Fs
freq_Hz = np.arange(0,int(nfft_samples/2+1),1)*Fs/nfft_samples

# Create the figure
fig = plt.figure(figsize = (16,5))
#---
plt.subplot(1,2,1)
plt.pcolormesh(taxis_spec,freq_Hz/1000, DOA_nb_SRP, cmap = 'jet')
plt.colorbar()
plt.title('Narrowband DOA with SRP')
#
# plt.subplot(1,2,2)
# plt.pcolormesh(taxis_spec,freq_Hz/1000, DOA_nb_MVDR, cmap = 'jet')
# plt.colorbar()
# plt.title('Narrowband DOA with MVDR')

#-----------FULLBAND FRAME-WISE ANGULAR SPECTRUM ----------

DOA_fb_SRP= np.sum(SRP[:,:,:],axis = 0)
maxvals = np.expand_dims(np.amax(DOA_fb_SRP,axis = 1),axis=1)
DOA_fb_SRP = np.divide(DOA_fb_SRP,maxvals)

DOA_fb_SRP_nonlin= np.sum(SRP_nonlin[:,:,:],axis = 0)
maxvals = np.expand_dims(np.amax(DOA_fb_SRP_nonlin,axis = 1),axis=1)
DOA_fb_SRP_nonlin = np.divide(DOA_fb_SRP_nonlin,maxvals)

# DOA_fb_MVDR= np.sum(MVDR[:,:,:],axis = 0)
# maxvals = np.expand_dims(np.amax(DOA_fb_MVDR,axis = 1),axis=1)
# DOA_fb_MVDR = np.divide(DOA_fb_MVDR,maxvals)


# Plot a few polar plots for several frames
# frames = [1,10, 11, 20]
# fig = plt.figure(figsize = (16,10))
# #---
# plt.subplot(221, projection = 'polar')
# plt.plot(scan_angles, DOA_fb_SRP[frames[0],:], color = 'b', label = 'SRP')
# plt.plot(scan_angles, DOA_fb_SRP_nonlin[frames[0],:], color = 'r', label = 'SRP_nonlin')
# plt.plot(scan_angles, DOA_fb_MVDR[frames[0],:], color = 'k', label = 'MVDR')
# plt.legend()
# plt.title('FB spectrum at different frames')
# #--
# plt.subplot(222, projection = 'polar')
# plt.plot(scan_angles, DOA_fb_SRP[frames[1],:], color = 'b', label = 'SRP')
# plt.plot(scan_angles, DOA_fb_SRP_nonlin[frames[1],:], color = 'r', label = 'SRP_nonlin')
# plt.plot(scan_angles, DOA_fb_MVDR[frames[1],:], color = 'k', label = 'MVDR')
# plt.legend()
# #--
# plt.subplot(223, projection = 'polar')
# plt.plot(scan_angles, DOA_fb_SRP[frames[2],:], color = 'b', label = 'SRP')
# plt.plot(scan_angles, DOA_fb_SRP_nonlin[frames[2],:], color = 'r', label = 'SRP_nonlin')
# plt.plot(scan_angles, DOA_fb_MVDR[frames[2],:], color = 'k', label = 'MVDR')
# plt.legend()
# #--
# plt.subplot(224, projection = 'polar')
# plt.plot(scan_angles, DOA_fb_SRP[frames[3],:], color = 'b', label = 'SRP')
# plt.plot(scan_angles, DOA_fb_SRP_nonlin[frames[3],:], color = 'r', label = 'SRP_nonlin')
# plt.plot(scan_angles, DOA_fb_MVDR[frames[3],:], color = 'k', label = 'MVDR')
# plt.legend()


#------ PLOT NARROWBAND ANGULAR SPECTRA AT SOME FREQUENCIES
# selectedFreqcs = [10, 50, 100, 200, 300, 400]
# frame = 10
#
# tmp_SRP= SRP[selectedFreqcs,frame,:]
# maxvals = np.expand_dims(np.amax(tmp_SRP,axis = 1),axis=1)
# tmp_SRP = np.divide(tmp_SRP,maxvals)
#
# tmp_SRP_nonlin= SRP_nonlin[selectedFreqcs,frame,:]
# maxvals = np.expand_dims(np.amax(tmp_SRP_nonlin,axis = 1),axis=1)
# tmp_SRP_nonlin = np.divide(tmp_SRP_nonlin,maxvals)
#
# tmp_MVDR= MVDR[selectedFreqcs,frame,:]
# maxvals = np.expand_dims(np.amax(tmp_MVDR,axis = 1),axis=1)
# tmp_MVDR = np.divide(tmp_MVDR,maxvals)


# # Plot a few polar plots for several frames
# fig = plt.figure(figsize = (16,15))
# #---
# plt.subplot(321, projection = 'polar')
# plt.plot(scan_angles, tmp_SRP[0,:], color = 'b', label = 'SRP')
# plt.plot(scan_angles, tmp_SRP_nonlin[0,:], color = 'r', label = 'SRP_nonlin')
# plt.plot(scan_angles, tmp_MVDR[0,:], color = 'k', label = 'MVDR')
# plt.legend()
# plt.title('NB spectrum, freqbin 10')
# #--
# plt.subplot(322, projection = 'polar')
# plt.plot(scan_angles, tmp_SRP[1,:], color = 'b', label = 'SRP')
# plt.plot(scan_angles, tmp_SRP_nonlin[1,:], color = 'r', label = 'SRP_nonlin')
# plt.plot(scan_angles, tmp_MVDR[1,:], color = 'k', label = 'MVDR')
# #plt.legend()
# plt.title('NB spectrum, freqbin 50')
# #--
# plt.subplot(323, projection = 'polar')
# plt.plot(scan_angles, tmp_SRP[2,:], color = 'b', label = 'SRP')
# plt.plot(scan_angles, tmp_SRP_nonlin[2,:], color = 'r', label = 'SRP_nonlin')
# plt.plot(scan_angles, tmp_MVDR[2,:], color = 'k', label = 'MVDR')
# #plt.legend()
# plt.title('NB spectrum, freqbin 100')
# #--
# plt.subplot(324, projection = 'polar')
# plt.plot(scan_angles, tmp_SRP[3,:], color = 'b', label = 'SRP')
# plt.plot(scan_angles, tmp_SRP_nonlin[3,:], color = 'r', label = 'SRP_nonlin')
# plt.plot(scan_angles, tmp_MVDR[3,:], color = 'k', label = 'MVDR')
# #plt.legend()
# plt.title('NB spectrum, freqbin 200')
# #-----
# plt.subplot(325, projection = 'polar')
# plt.plot(scan_angles, tmp_SRP[4,:], color = 'b', label = 'SRP')
# plt.plot(scan_angles, tmp_SRP_nonlin[4,:], color = 'r', label = 'SRP_nonlin')
# plt.plot(scan_angles, tmp_MVDR[4,:], color = 'k', label = 'MVDR')
# #plt.legend()
# plt.title('NB spectrum, freqbin 300')
# #-----
# plt.subplot(326, projection = 'polar')
# plt.plot(scan_angles, tmp_SRP[5,:], color = 'b', label = 'SRP')
# plt.plot(scan_angles, tmp_SRP_nonlin[5,:], color = 'r', label = 'SRP_nonlin')
# plt.plot(scan_angles, tmp_MVDR[5,:], color = 'k', label = 'MVDR')
# #plt.legend()
# plt.title('NB spectrum, freqbin 400')
#


#--------- BATCH ANGULAR SPECTRUM: sum function and max function

# Sum across all times and frequencies
DOA_batch_SRP = np.sum(np.sum(SRP[:,:,:],axis = 0),axis=0)
DOA_batch_SRP = DOA_batch_SRP/np.amax(DOA_batch_SRP)

DOA_batch_SRP_nonlin = np.sum(np.sum(SRP_nonlin[:,:,:],axis = 0),axis=0)
DOA_batch_SRP_nonlin = DOA_batch_SRP_nonlin/np.amax(DOA_batch_SRP_nonlin)

#DOA_batch_MVDR = np.sum(np.sum(MVDR[:,:,:],axis = 0),axis=0)
#DOA_batch_MVDR =  DOA_batch_MVDR/np.amax(DOA_batch_MVDR)


# Do not sum across time but take the maximum
DOA_batch_SRP_max = np.amax(np.sum(SRP,axis=0),axis =0)
DOA_batch_SRP_max = DOA_batch_SRP_max/np.amax(DOA_batch_SRP_max)

DOA_batch_SRP_nonlin_max = np.amax(np.sum(SRP_nonlin,axis=0),axis =0)
DOA_batch_SRP_nonlin_max = DOA_batch_SRP_nonlin_max/np.amax(DOA_batch_SRP_nonlin_max)

#DOA_batch_MVDR_max = np.amax(np.sum(MVDR,axis=0),axis =0)
#OA_batch_MVDR_max =  DOA_batch_MVDR_max/np.amax(DOA_batch_MVDR_max)

fig = plt.figure(figsize = (15,6))
#---
plt.subplot(121, projection = 'polar')
plt.plot(scan_angles, DOA_batch_SRP, color = 'b', label = 'SRP',linestyle = ':')
plt.plot(scan_angles, DOA_batch_SRP_nonlin, color = 'r', label = 'SRP_nonlin' ,linestyle = '--')
#plt.plot(scan_angles, DOA_batch_MVDR, color = 'g', label = 'MVDR' ,linestyle = '-')
plt.legend()
plt.title('Batch fullband spectrum')
#---
plt.subplot(122, projection = 'polar')
plt.plot(scan_angles, DOA_batch_SRP_max, color = 'b', label = 'SRP',linestyle = ':')
plt.plot(scan_angles, DOA_batch_SRP_nonlin_max, color = 'r', label = 'SRP_nonlin' ,linestyle = '--')
#plt.plot(scan_angles, DOA_batch_MVDR_max, color = 'g', label = 'MVDR' ,linestyle = '-')
plt.legend()

plt.show()

#fig = plt.figure(figsize=(8,8))
#ax = plt.subplot(111, projection= 'polar')
#ax.plot(scan_angles, output, color = 'b', label = 'full',linestyle = ':')
#ax.plot(scan_angles, output_l, color = 'r', label = 'low' ,linestyle = '--')
#ax.plot(scan_angles, output_h, color = 'g', label = 'high' ,linestyle = '-')
#ax.legend()
