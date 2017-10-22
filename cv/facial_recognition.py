import cv2
import imutils
import predictor

print("Running...")

cap = cv2.VideoCapture('../data/James.mp4')
# cap = cv2.VideoCapture(0)

p1 = predictor.Predictor()
frame_interval = predictor.runDiagnostic()

while (cap.isOpened()):
    ret, frame = cap.read()

    counter = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        rs = imutils.resize(frame,width=500)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # resized_gray = imutils.resize(gray, width=500)
        # cv2.imshow('frame', resized_gray)
        counter += 1
        if counter%15 == 0:
            p1.next([frame])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            counter = 0
        cv2.imshow("color",rs)
            #cv2.imshow('color', mouths)



cap.release()
