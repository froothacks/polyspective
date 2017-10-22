import cv2
import numpy as np
import imutils


cap = cv2.VideoCapture('../data/Rio.mp4')


class Predicter:
    def __init__(self):
        self.prevFrame = None

    def next(self,img):
        frame = imutils.resize(img, width=500)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (21, 21), 0)


while cap.isOpened():
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=500)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (21, 21), 0)
    # if the first frame is None, initialize it
    if prevFrame is not None:
        frameDelta = cv2.absdiff(img, prevFrame)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        change = np.sum(thresh**2)
        print(change)
        cv2.imshow("Original", frame)
        cv2.imshow("thresh", thresh)
        cv2.imshow("FrameDelta", frameDelta)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    prevFrame = img

