
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import time

class constants:
    DATA_PATH="../data"
    TEST_PATH=DATA_PATH+"/test_data"
    LANDMARK_TRAINING_SET = "shape_predictor_68_face_landmarks.dat"
    FULLY_OPEN = 0.914772727275
    FULLY_CLOSED = 0.0824213743466


class utils:
    @staticmethod
    def rect_to_bb(rect):
        # take a bounding predicted by dlib and convert it
        # to the format (x, y, w, h) as we would normally do
        # with OpenCV
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
     
        # return a tuple of (x, y, w, h)
        return (x, y, w, h)
    @staticmethod
    def shape_to_np(shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)
     
        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
     
        # return the list of (x, y)-coordinates
        return coords
    @staticmethod
    def getMouthPoints(landmarks):
        mouth = {
            "outer_lips" : landmarks[49-1:60+1-1],
            "inner_lips" : landmarks[61-1:68+1-1],
        }
        return mouth

class algorithms:
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = None
    predictor = None
    @staticmethod
    def init():
        algorithms.detector = dlib.get_frontal_face_detector()
        algorithms.predictor = dlib.shape_predictor(constants.DATA_PATH+"/"+constants.LANDMARK_TRAINING_SET)
    @staticmethod
    def getFacePoints(img,debug=False):
        # my own sort of 'fork' of this website's code https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python

        ## >>>THE REAL CODE STARTS HERE <<<
        if debug: print("Reading the testing image...");t = time.time()
        # load the input image, resize it, and convert it to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("",gray)
        #cv2.waitKey(0)
        if debug: print("Finished. %s seconds elapsed" % str(time.time()-t) )
        if debug: print("Detecting faces...");t = time.time()
        # detect faces in the grayscale image
        rects = algorithms.detector(gray, 1)
        if len(rects) < 1:
            ##print("none")
            return []
        if debug: print("Finished. %s seconds elapsed" % str(time.time()-t) )
        face_set = []
        for rect in rects:
            face_set.append(
                np.matrix([
                    [p.x, p.y] for p in algorithms.predictor(gray, rect).parts()
                ])
            )
        return face_set
    @staticmethod
    def getMouthOpen(landmarks,debug=False):
        # locating all the points from our list of landmarks
        base_left = landmarks[61-1] # inner mouth, far left
        base_right = landmarks[65-1] # inner mouth, far right
        top_left = landmarks[62-1]
        bottom_left = landmarks[68-1]
        top_right = landmarks[64-1]
        bottom_right = landmarks[66-1]
        # calculating slopes
        lt = abs((base_left[0,1]  - top_left[0,1])     / (base_left[0,0]  - top_left[0,0])    )
        lb = abs((base_left[0,1]  - bottom_left[0,1])  / (base_left[0,0]  - bottom_left[0,0]) )
        rt = abs((base_right[0,1] - top_right[0,1])    / (base_right[0,0] - top_right[0,0])   )
        rb = abs((base_right[0,1] - bottom_right[0,1]) / (base_right[0,0] - bottom_right[0,0]))
        # think of this constants.FULLY_OPEN like a "perfect score" (100%)
        # divide the average of the by constants.FULLY_OPEN to become a score from 0.0 - 1.0
        # (think of it like 0% to 100%)
        score = max(((lt+lb+rt+rb)/4)-constants.FULLY_CLOSED,0)/constants.FULLY_OPEN
        return score

def runDiagnostic():
    image = cv2.imread(constants.TEST_PATH+"/cw.jpg")
    t = time.time()
    image = imutils.resize(image, width=500)
    landmarks = algorithms.getFacePoints(image)
    diff = time.time() - t
    FPS = 30
    GOAL_TIME = 0.5
    # sgg is the suggested interval for frame grabbing
    sgg = int((diff*FPS)/GOAL_TIME)
    return sgg
def getMouthsFromFaceSet(landmarks):
    s = []
    ##print(len(landmarks))
    for i in landmarks:
        s.append(utils.getMouthPoints(i))
    return s
def getTestingImage(img):
    img = imutils.resize(img,width=500)
    face_set = algorithms.getFacePoints(img)
    mouth_score = algorithms.getMouthOpen(face_set[0])
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(img, str(mouth_score), (100, 100), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
    mouths = getMouthsFromFaceSet(face_set)
    for mouth in mouths:
        for coords in mouth["inner_lips"]:
            cv2.circle(img, (coords[0,0], coords[0,1]), 1, (255, 0, 0), thickness=-1)
    return img
def main(debug=False):
    algorithms.init()
    if debug:
        print("Running...")
        t = time.time()
    image = cv2.imread(constants.DATA_PATH+"/test_data/cw2.jpg")
    img = getTestingImage(image)
    cv2.imshow("",img)
    cv2.waitKey(0)
if __name__ == '__main__':
    main(debug=True)
