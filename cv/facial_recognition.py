import cv2
import imutils

cap = cv2.VideoCapture('video.mp4')

while (cap.isOpened()):
    ret, frame = cap.read()
    while (cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_gray = imutils.resize(gray, width=500)
        cv2.imshow('frame', resized_gray)
        resized_color = imutils.resize(frame, width=500)
        cv2.imshow('color', resized_color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
