import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift
import math
from scipy.signal import blackman, hamming, hanning, cosine, blackmanharris


#----------analysis filterbank: forward short-time Fourier transform

def stft(sig_in,winsize_samples,nfft_samples,overlap_percentage,wintype):

    hopsize_fraction = (100-overlap_percentage)/100
    hopsize = int(hopsize_fraction*winsize_samples)

    if wintype == 'cosine':
        win_analysis = cosine(winsize_samples,sym=False)
    elif wintype == 'blackmanharris':
        win_analysis = blackmanharris(winsize_samples,sym=False)
    elif wintype == 'blackman':
        win_analysis = blackman(winsize_samples,sym=False)
    elif wintype == 'hamming':
        win_analysis = hamming(winsize_samples,sym=False)
    elif wintype == 'hanning':
        win_analysis = hanning(winsize_samples,sym=False)


    # to make sure we don't run out of samples at the end of the signal
    numFrames = math.ceil(sig_in.size/hopsize)-int(1/hopsize_fraction)

    # store only half of the spectrum (the other half is the complex conjugate)
    specsize = int(nfft_samples/2 + 1)

    stft_spectrum = np.zeros(shape = (specsize,numFrames))
    stft_spectrum = stft_spectrum + 0.j

    # Transform the signal frame-by-frame and store the spectrum
    for idx in range(0,numFrames):
        startid = int(idx*hopsize)
        endid = startid + winsize_samples
        currentFrame = sig_in[startid:endid]

        # if nfft_samples>win_analysis.size, the frame is zero-padded before FFT
        tmpspec = fft(win_analysis*currentFrame,nfft_samples)
        stft_spectrum[:,idx] = tmpspec[0:specsize]

    return stft_spectrum



#----------synthesis filterbank: inverse short-time Fourier transform

def istft(spectrum,winsize_samples,nfft_samples,overlap_percentage,wintype):

    hopsize_fraction = (100-overlap_percentage)/100
    hopsize = int(hopsize_fraction*winsize_samples)
    specsize = int(nfft_samples/2 + 1)

    # get the synthesis window
    if wintype == 'cosine':
        win_synthesis = cosine(winsize_samples,sym=False)
    elif wintype == 'blackmanharris':
        win_synthesis = blackmanharris(winsize_samples,sym=False)
    elif wintype == 'blackman':
        win_synthesis = blackman(winsize_samples,sym=False)
    elif wintype == 'hamming':
        win_synthesis = hamming(winsize_samples,sym=False)
    elif wintype == 'hanning':
        win_synthesis = hanning(winsize_samples,sym=False)

    #make sure that the scaling is correct
    sqsum = np.sum(np.square(win_synthesis))/winsize_samples
    invsqsum = 1/sqsum
    tmp = hopsize_fraction
    correctionFactor = tmp*invsqsum
    win_synthesis = win_synthesis*correctionFactor

    # allocate memory for the signal
    numFrames = spectrum.shape[1]
    framesize = int((100-overlap_percentage)*winsize_samples/100)

    # allocate a 2D array for the time-domain signal (to be vectorized later)
    sig_hat = np.zeros(shape = (numFrames,framesize))

    # prepare buffers for the istft
    buffer_chunks = int(1/hopsize_fraction)
    ifft_buffer = np.zeros(shape = (winsize_samples,buffer_chunks))


    for idx in range(0,numFrames):

        # get the current frame spectrum and append the complex conjugate
        tmpspec = np.zeros(shape = (nfft_samples)) + 0.j
        tmpspec[0:specsize] = spectrum[:,idx] # select the column of the spectrogram
        tmpspec[specsize:nfft_samples] = np.conj(np.flip(spectrum[1:specsize-1,idx],0))

        # take the ifft of the spectrum
        tmpframe = np.real(ifft(tmpspec,nfft_samples))

        # apply the synthesis window to the time frame, and keep only the number of samples corresponding to the analysis window (if zero-padding was applied, this ensures that the irrelevant samples are discarded)
        frame_in_time = win_synthesis*tmpframe[0:winsize_samples]

        # put the last one out, take the new one in
        ifft_buffer[:,1:buffer_chunks] = ifft_buffer[:,0:buffer_chunks-1]
        ifft_buffer[:,0] = frame_in_time

        # implementation of the overlap-add procedure
        frame_out = np.zeros(shape = (hopsize))
        for bufidx in range(0,buffer_chunks):
            bufstart = bufidx*hopsize
            bufend = bufstart + hopsize
            frame_out = frame_out + ifft_buffer[bufstart:bufend,bufidx]

        # store the current output frame in the correct indices of the signal
        sig_hat[idx,:] = frame_out

        #--- end of for idx in range(0,numFrames)

    # vectorize 2D signal to obtain one time-domain signal vector
    sig_out =  sig_hat.flatten()

    return sig_out
