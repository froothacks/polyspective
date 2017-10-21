import cv2
import imutils

from cv.Predictor import algorithms, utils

cap = cv2.VideoCapture('../data/video.mp4')

while (cap.isOpened()):
    ret, frame = cap.read()
    algorithms.init()
    while (cap.isOpened()):
        ret, frame = cap.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # resized_gray = imutils.resize(gray, width=500)
        # cv2.imshow('frame', resized_gray)
        resized_color = imutils.resize(frame, width=500)
        landmarks = algorithms.getFacePoints(resized_color)

        if type(landmarks) != type(None):
            # print (landmarks)
            print("faces found!")
            mouth = utils.getMouthPoints(landmarks)

            for coords in mouth["outer_lips"]:
                cv2.circle(resized_color, (coords[0, 0], coords[0, 1]), 2, (255, 0, 0), thickness=-1)
        cv2.imshow('color', resized_color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()