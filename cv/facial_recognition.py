import cv2

import predictor

cap = cv2.VideoCapture('../data/James.mp4')
# cap = cv2.VideoCapture(0)

# will change this later to just Predictor.init() when Calder updates code
predictor.Algorithms.init()
frame_interval = predictor.runDiagnostic()

while (cap.isOpened()):
    ret, frame = cap.read()

    counter = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # resized_gray = imutils.resize(gray, width=500)
        # cv2.imshow('frame', resized_gray)
        counter += 1
        if counter == int(frame_interval/2):
            mouths = predictor.getTestingImage(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            counter = 0
            cv2.imshow('color', mouths)



cap.release()
