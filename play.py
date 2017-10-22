import cv2
from cv import predictor
import numpy as np
#laptop
cl = cv2.VideoCapture(0)
#leon
#c = cv2.VideoCapture('http://100.65.194.194:4747/mjpegfeed')
#ad
c = cv2.VideoCapture('http://100.64.228.178:4747/mjpegfeed')
# while(1):
#     _,f = c.read()
#     cv2.imshow('e2',f)
#     _, fl = cl.read()
#     cv2.imshow('e3', fl)
#     if cv2.waitKey(5)==27:
#         break
# cv2.destroyAllWindows()
p1 = predictor.Predictor()
# predictor.Algorithms.init()
frame_interval = predictor.runDiagnostic()
counter = 0
lastf = 0
while (1):
    # ret, frame = cap.read()
    _, f = c.read()
    #cv2.imshow('e2', f)
    _, fl = cl.read()
    #cv2.imshow('e3', fl)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # resized_gray = imutils.resize(gray, width=500)
    # cv2.imshow('frame', resized_gray)
    counter += 1
    #print("here")
    fs = [fl, f]
    #cv2.imshow("stream", lastf)
    if counter % frame_interval/3 == 0:
        ret = p1.next(fs)
        if max(ret) == 0:
            print("ZERO")
            cv2.imshow("stream", fs[lastf])
        else:
            best = fs[ret.index(max(ret))]
            cv2.imshow("stream", best)
            lastf = ret.index(max(ret))
        print(ret)
        print(ret.index(max(ret)))
        #mouths = predictor.getTestingImage(f)
        #mouths2 = predictor.getTestingImage(fl)
    
    else:
        cv2.imshow("stream", fs[lastf])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #cv2.imshow('color', mouths)
    #cv2.imshow('color2', mouths2)



# cap.release()
