#!/usr/bin/env python3
# CHANGE THIS LINE TO USE OTHER PREDICTION METHOD
from ml import predictor
# from cv import predictor

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ml.preprocessor import getAudioData, getFFT

fname = "data/masquerade"

audio, audioSampleRate = getAudioData(fname, True)

print("AUDIO DATA INFORMATION")
print("NUM SAMPLES:", audio.shape)
print("SAMPLE RATE:", audioSampleRate)
print("MIN:", np.amin(audio))
print("MAX:", np.amax(audio))
print("AVG:", np.average(audio))

cap = cv2.VideoCapture("%s.mov" % fname)

while(cap.isOpened()):
    ret, frame = cap.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
