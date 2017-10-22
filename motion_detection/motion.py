import cv2
import numpy as np
import imutils


class Predictor:
    def __init__(self, num_cams):
        self.num_cams = num_cams
        self.prevFrame = [None] * self.num_cams

    def next(self, frames):
        scores = []
        for i in range(len(frames)):
            img = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            img = cv2.GaussianBlur(img, (21, 21), 0)
            if (self.prevFrame[i]) is not None:
                frameDelta = cv2.absdiff(img, self.prevFrame[i])
                cv2.imshow("frameDelta", frameDelta)
                change = np.sum(frameDelta ** 2)
                scores.append(change)

            self.prevFrame[i] = img

        if len(self.prevFrame) > self.num_cams:
            self.prevFrame.pop(0)
            self.prevFrame.pop(1)

        scores = [s / (self.prevFrame[i].size * 128) for i,s in enumerate(scores)]
        print("SCOROEOEE")
        print(scores)
        return scores


#
if __name__ == '__main__':
    pMotion = Predictor(2)
    cap = cv2.VideoCapture('../data/Rio.mp4')
    cap2 = cv2.VideoCapture(0)
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     frame = imutils.resize(frame, width=500)
    #     img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     img = cv2.GaussianBlur(img, (21, 21), 0)
    #     # if the first frame is None, initialize it
    #     if prevFrame is not None:
    #         frameDelta = cv2.absdiff(img, prevFrame)
    #         thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    #         change = np.sum(thresh**2)
    #         print(change)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #
    #     prevFrame = img
    while cap.isOpened():
        ret, frame = cap.read()
        ret2, frame2 = cap2.read()
        cv2.imshow("frame1", imutils.resize(frame, width=200))
        cv2.imshow("frame2", frame2)
        print(pMotion.next([frame, frame2]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
