import cv2

from cv import Predictor

cap = cv2.VideoCapture('../data/James.mp4')
# cap = cv2.VideoCapture(0)

# will change this later to just Predictor.init() when Calder updates code
Predictor.algorithms.init()
frame_interval = Predictor.runDiagnostic()

while (cap.isOpened()):
    ret, frame = cap.read()

    counter = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # resized_gray = imutils.resize(gray, width=500)
        # cv2.imshow('frame', resized_gray)
        counter += 1
        if counter == 1:
            mouths = Predictor.getTestingImage(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            counter = 0
            cv2.imshow('color', mouths)



cap.release()
