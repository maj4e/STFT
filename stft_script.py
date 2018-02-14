#!/Users/mtaseska/Python_Projects/STFT/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq, fftshift
import math
from scipy.signal import blackman, hamming, hanning, cosine, blackmanharris
import soundfile as sf
from scipy.io.wavfile import write as wavwrite


pi = math.pi


# %% Reading, windowing and fft-transforming the audio signal
sig1, fs = sf.read('/Users/mtaseska/AudioData/samples_speech/female_16.wav')
sig1 = sig1[0:150000]


# Filterbank parameters
analysis_win_len_samples = 1024; # in samples

nfft_size = analysis_win_len_samples # with zeropadding here

zeropad_samples = nfft_size - analysis_win_len_samples
analysis_win_len_seconds = analysis_win_len_samples/fs

hopsize_fraction = 0.5
buffer_chunks = int(1/hopsize_fraction)
hopsize = int(hopsize_fraction*analysis_win_len_samples)
ifft_buffer = np.zeros(shape = (analysis_win_len_samples,buffer_chunks))

win_analysis = cosine(analysis_win_len_samples ,sym=False)
win_synthesis = win_analysis

numFrames = math.ceil(sig1.size/hopsize)-8
specsize = int(nfft_size/2 + 1)

# allocate the spectrogram
Spectro = np.zeros(shape = (specsize,numFrames))
Spectro = Spectro + 0.j

# The forward transform
for idx in range(0,numFrames):
    startid = int(idx*hopsize)
    endid = startid + analysis_win_len_samples
    currentFrame = sig1[startid:endid]
    tmpspec = fft(win_analysis*currentFrame,nfft_size)
    Spectro[:,idx] = tmpspec[0:specsize]


# The backward transform
sig_hat = np.zeros(shape = (sig1.size))
idx = 0
for idx in range(0,numFrames):

    startid = int(idx*hopsize)
    endid = startid + hopsize

    tmpspec = np.zeros(shape = (nfft_size)) + 0.j
    tmpspec[0:specsize] = Spectro[:,idx] # select the column of the spectrogram
    tmpspec[specsize:nfft_size] = np.conj(np.flip(Spectro[1:specsize-1,idx],0))

    tmpframe = np.real(ifft(tmpspec,nfft_size))
    frame_in_time = win_synthesis*tmpframe[0:analysis_win_len_samples]

    # put the last one out, take the new one in
    ifft_buffer[:,1:buffer_chunks] = ifft_buffer[:,0:buffer_chunks-1]
    ifft_buffer[:,0] = frame_in_time

    frame_out = np.zeros(shape = (hopsize))

    for bufidx in range(0,buffer_chunks):
        bufstart = bufidx*hopsize
        bufend = bufstart + hopsize
        frame_out = frame_out + ifft_buffer[bufstart:bufend,bufidx]

    sig_hat[startid:endid] = frame_out


taxis_spec = np.arange(0,numFrames,1)*hopsize/fs
taxis_time = np.arange(0,sig1.size,1)/fs
faxis = np.arange(0,specsize,1)


# The two time-domain signals
fig, axes = plt.subplots(2,1,figsize = (14,10))
axes[0].plot(taxis_time,sig_hat)
axes[1].plot(taxis_time,sig1)
wavwrite('output.wav',fs,sig_hat)


# The forward transform of the reconstructed signal
Spectro_hat = np.zeros(shape = (specsize,numFrames))
Spectro_hat = Spectro_hat + 0.j
for idx in range(0,numFrames):
    startid = int(idx*hopsize)
    endid = startid + analysis_win_len_samples
    currentFrame = sig_hat[startid:endid]
    tmpspec = fft(win_analysis*currentFrame,nfft_size)
    Spectro_hat[:,idx] = tmpspec[0:specsize]



fig, axes = plt.subplots(2,1,figsize = (14,10))
axes[0].pcolormesh(taxis_spec,faxis, 10*np.log10(np.abs(Spectro)), cmap = 'PRGn')
axes[1].pcolormesh(taxis_spec,faxis, 10*np.log10(np.abs(Spectro_hat)), cmap = 'PRGn')
#axes[1].plot(taxis_time,sig1)
#axes[1].set_xlim([0,taxis_time[taxis_time.size-1]])
plt.show()
