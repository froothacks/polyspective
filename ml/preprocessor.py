import os
import subprocess
from warnings import warn

from scipy.fftpack import fft
from scipy.io import wavfile
import numpy as np

def getAudioData(fname, force_refresh=False):
    if force_refresh or not os.path.isfile("%s_audio.wav" % fname):
        print("wav file doesn't exist, creating now")
        cmd = 'ffmpeg -i "%s.mov" -ac 1 -sample_fmt s16 -vn -y -nostats -loglevel 0 "%s_audio.wav"' % (fname, fname)
        subprocess.run(cmd, shell=True)

    rate, data = wavfile.read("%s_audio.wav" % fname)

    # Normalize data to interval (-1, 1)
    data = data / (2**15)

    if np.amin(data) < -1 or (np.amax(data)) > 1:
        warn("Audio stream for file %s exceeds max amplitude")
    if abs(np.average(data)) > 0.01:
        warn("Audio stream imbalanced - centered on %f" % np.average(audio))

    return data, rate

def getFFT(data):
    s = fft(data)
    s = s[:len(s)//2] # We only need the first half (real signal symmetry)
    s = np.absolute(s) # Get amplitude
    return s