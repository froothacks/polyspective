from numpy import matrix as np_matrix
import time
import random
import imutils
import dlib
import cv2


class Constants:
    DATA_PATH = "../data"
    TEST_PATH = DATA_PATH + "/test_data"
    LANDMARK_TRAINING_SET = "shape_predictor_68_face_landmarks.dat"
    FULLY_OPEN = 0.914772727275
    FULLY_CLOSED = 0.0824213743466
    RESISZE_WIDTH = 350

class Algorithms:
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = None
    predictor = None

    @staticmethod
    def init():
        Algorithms.detector = dlib.get_frontal_face_detector()
        Algorithms.predictor = dlib.shape_predictor(Constants.DATA_PATH + "/" + Constants.LANDMARK_TRAINING_SET)

    @staticmethod
    def getFacePoints(img, debug=False):
        # my own sort of 'fork' of this website's code https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python

        ## >>>THE REAL CODE STARTS HERE <<<
        if debug: print("Reading the testing image...");t = time.time()
        # load the input image, resize it, and convert it to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("",gray)
        # cv2.waitKey(0)
        if debug: print("Finished. %s seconds elapsed" % str(time.time() - t))
        if debug: print("Detecting faces...");t = time.time()
        # detect faces in the grayscale image
        rects = Algorithms.detector(gray, 1)
        if len(rects) < 1:
            ##print("none")
            return []
        if debug: print("Finished. %s seconds elapsed" % str(time.time() - t))
        face_set = []
        for rect in rects:
            face_set.append(
                np_matrix([
                    [p.x, p.y] for p in Algorithms.predictor(gray, rect).parts()
                ])
            )
        return face_set

    @staticmethod
    def getMouthOpen(landmarks, debug=False):
        # locating all the points from our list of landmarks
        base_left = landmarks[61 - 1]  # inner mouth, far left
        base_right = landmarks[65 - 1]  # inner mouth, far right
        top_left = landmarks[62 - 1]
        bottom_left = landmarks[68 - 1]
        top_right = landmarks[64 - 1]
        bottom_right = landmarks[66 - 1]
        # calculating slopes
        lt = min(abs((base_left[0, 1] - top_left[0, 1]) / max(base_left[0, 0] - top_left[0, 0],1)),1)
        lb = min(abs((base_left[0, 1] - bottom_left[0, 1]) / max(base_left[0, 0] - bottom_left[0, 0],1)),1)
        rt = min(abs((base_right[0, 1] - top_right[0, 1]) / max(base_right[0, 0] - top_right[0, 0],1)),1)
        rb = min(abs((base_right[0, 1] - bottom_right[0, 1]) / max(base_right[0, 0] - bottom_right[0, 0],1)),1)
        # substact the face's slope from each slope
        face_slope = abs(Algorithms.getFaceSlope(landmarks))
        lt -= face_slope
        lb -= face_slope
        rt -= face_slope
        rb -= face_slope
        # think of this constants.FULLY_OPEN like a "perfect score" (100%)
        # divide the average of the by constants.FULLY_OPEN to become a score from 0.0 - 1.0
        # (think of it like 0% to 100%)
        score = max(((lt + lb + rt + rb) / 4) - Constants.FULLY_CLOSED, 0) / Constants.FULLY_OPEN
        return score

    @staticmethod
    def getFaceSlope(landmarks):
        """Using 2 points for 2 eyes, and sohcahtoa this finds the tilt of the face."""
        # act as if the point is a point on the center's circumference
        a, b = landmarks[40 - 1], landmarks[43 - 1]
        return (a[0, 1] - b[0, 1]) / (a[0, 0] - b[0, 1])


def runDiagnostic():
    image = cv2.imread(Constants.TEST_PATH + "/cw.jpg")
    t = time.time()
    image = imutils.resize(image, width=Constants.RESISZE_WIDTH)
    landmarks = Algorithms.getFacePoints(image)
    diff = time.time() - t
    FPS = 30
    GOAL_TIME = 0.5
    # sgg is the suggested interval for frame grabbing
    sgg = int((diff * FPS) / GOAL_TIME)
    return sgg


def getTestingImage(img):
    img = imutils.resize(img, width=Constants.RESISZE_WIDTH)
    face_set = Algorithms.getFacePoints(img)
    font = cv2.FONT_HERSHEY_PLAIN
    ypos = 100
    def r(a):return (255+(1 - 2*random.randint(0,1))*(a*50))%255
    colors = [(r(i),r(i),r(i)) for i in range(len(face_set))]
    for place,fs in enumerate(face_set):
        R,G,B = colors[place]
        mouth_score = Algorithms.getMouthOpen(fs)
        cv2.putText(img, "MS: " + str(mouth_score), (100, ypos), font, 1, (R, G, B), 1, cv2.LINE_AA)
        ypos += 20
        for coords in fs:
            cv2.circle(img, (coords[0, 0], coords[0, 1]), 1, (R, G, B), thickness=-1)

    return img

class Face():
    def __init__(self,img):
        """A class to score a face."""
        self.landmarks = Algorithms.getFacePoints(img)
        ##if len(self.landmarks) > 1:
        ##    print("Warning! More than 1 face found.")
    def get_mouth_score(self,landmark_index):
        """Calder's function to return a score from 0.0. to 1.0 on how open the face's mouth is."""
        return Algorithms.getMouthOpen(self.landmarks[landmark_index])

## THIS IS THE PUBLICLY USED FUNCTION
class Predictor(object):
    """The public class for use in the main program."""
    def __init__(self):
        Algorithms.init()
        self.open_past = []
        self.MAX_FRAMES = 10
    def update_opens(self,scores):
        if len(self.open_past) < len(scores):
            for i in range(len(scores)):
                self.open_past.append([])
        for i in range(len(scores)):
            self.open_past[i].append(scores[i])

            if len(self.open_past[i]) > self.MAX_FRAMES:
                self.open_past[i] = self.open_past[i][-self.MAX_FRAMES:]
    def next(self,frames):
        scores = [0.0]*len(frames)
        # first scoring the current face
        for place,frame in enumerate(frames):
            fc = Face(frame)
            opens = []
            for f in range(len(fc.landmarks)):
                opens.append(
                    fc.get_mouth_score(f)
                )
            if len(opens)>0:scores[place] = max(opens)
        self.update_opens(scores)
        rscores = []
        for i in range(len(frames)):
            # now interpolate zero opens
            for place,pt in enumerate(self.open_past[i]):
                if pt == 0.0:
                    if place-1 > 1 and place+1 < len(self.open_past[i]):
                        pt = (self.open_past[i][place-1]+self.open_past[i][place+1])/2
                        self.open_past[i][place]=pt
            # finally, convert hour graph of open_past in a movement difference
            diffs = []
            #print(len(self.open_past[i]))
            for j in range(len(self.open_past[i])-1):
                diffs.append(
                    abs(self.open_past[i][j-1]-self.open_past[i][j+1])
                )
            rscores.append(sum(diffs)/max(len(diffs),1))
            #print(diffs)
        return rscores
def main():
    print("Loading...")
    im = cv2.imread(Constants.DATA_PATH + "/test_data/cw.jpg")
    im2 = cv2.imread(Constants.DATA_PATH + "/test_data/cw2.jpg")
    ##img = getTestingImage(image)
    ##cv2.imshow("", img)
    ##cv2.waitKey(0)
    p = Predictor()
    print("Running...")
    p.next([im,im2])


if __name__ == '__main__':
    main()