# Checking the constant-overlap-add property of a window

# For a window pair to be used as analysis-synthesis pair, the composite window win_analysis*win_synthesis needs to satisfy the COLA property.

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift
import math
from scipy.signal import blackman, hamming, hanning, cosine, blackmanharris

num_windows = 10

# define the stft filterbank parameters
winsize_samples = 1024
overlap_percentage = 79
hop = int(winsize_samples*(100-overlap_percentage)/100)

totlen = (num_windows+5)*hop
allwindows = np.zeros(shape = (totlen,num_windows))

mywin = np.square(blackman(winsize_samples,sym=False))
sqsum = np.sum(mywin)/winsize_samples
invsqsum = 1/sqsum
tmp = (100-overlap_percentage)/100
correctionFactor = tmp*invsqsum


for idx in range(0,num_windows):
    startsample = idx*hop
    endsample = idx*hop + winsize_samples
    tmp = np.zeros(shape = (totlen))
    tmp[startsample:endsample] = mywin
    allwindows[:,idx] = tmp

output = np.sum(allwindows,1)*correctionFactor

fig, axes = plt.subplots(2,1,figsize = (14,5))
axes[0].plot(allwindows[:,0])
axes[0].plot(allwindows[:,1])
axes[0].plot(allwindows[:,2])
axes[0].plot(allwindows[:,3])
axes[0].plot(allwindows[:,4])
axes[0].plot(allwindows[:,5])
axes[0].plot(allwindows[:,6])
axes[0].plot(allwindows[:,7])
axes[0].plot(allwindows[:,8])
axes[0].plot(allwindows[:,9])
axes[0].plot(output,'+')
