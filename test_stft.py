# import the necessary packages
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.io.wavfile import write as wavwrite

# import my stft module
from module_stft import *


# load the signal to be analyzed
sig_in, fs = sf.read('/Users/mtaseska/AudioData/samples_speech/female_16.wav')
sig_in = sig_in[0:150000]

# define the stft filterbank parameters
winsize_samples = 1024
nfft_samples = 1024
overlap_percentage = 80
wintype = 'cosine'

# call the stft function from the module to transform the signal to the frequency domain
spectrum = stft(sig_in,winsize_samples,nfft_samples,overlap_percentage,wintype)

# call the istft function from the module to transform the signal back to time domain
sig_out = istft(spectrum,winsize_samples,nfft_samples,overlap_percentage,wintype)

# Cut the input signal so that the input and output signals match
sig_in = sig_in[0:sig_out.shape[0]]

# Plot the spectrum that we obtained
hopsize = int((100-overlap_percentage)*winsize_samples/100)
taxis_spec = np.arange(0,spectrum.shape[1],1)*hopsize/fs
taxis_time = np.arange(0,sig_in.size,1)/fs
faxis = np.arange(0,spectrum.shape[0],1)*fs/nfft_samples

#fig, axes = plt.subplots(2,1,figsize = (14,10))
#axes[0].pcolormesh(taxis_spec,faxis, 10*np.log10(np.abs(spectrum)), cmap = 'PRGn')

# The two time-domain signals
fig, axes = plt.subplots(2,1,figsize = (14,10))
axes[0].plot(sig_out,'r')
axes[1].plot(sig_in)

wavwrite('input.wav',fs,sig_in)
wavwrite('output.wav',fs,sig_out)
