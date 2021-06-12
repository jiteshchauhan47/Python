import cvlib as cv
from cvlib.object_detection import draw_bbox
import sys
import cv2
import numpy as np
from threading import Thread, Lock, Event
import time

g_bUseWebcam = True
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (0, 255, 0)
thickness = 2
org = (50, 50)

g_frame = np.zeros((480, 640, 3), np.uint8)

mutexDetection = Lock()
mutexDraw = Lock()
g_event = Event()

class BoundingBoxes:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.type = ""
    def __init__(self,x,y,w,h,type):
        self.x = x
        self.y = y
        self.width = w
        self.height = h
        self.type = type
    def rect(self):
        return (self.x, self.y, self.width, self.height)

class ObjectDetection:
    def __init__(self, frame):
        self.objectbox = []
        self.frame = frame
        self.bIsAppExitCalled = False;
    def infer(self,e,cameraNumber):
        try:
            while self.bIsAppExitCalled == False:     
                if(e.wait() == True):    
                    mutexDetection.acquire(blocking=True)
                    bbox, label, conf = cv.detect_common_objects(self.frame, 0.5, 0.3, 'yolov4-tiny', False)
                    mutexDetection.release()
                    print(" ::cameraNumber : "+ str(cameraNumber) + ":: "+ str(len(bbox)))
                    self.objectbox.clear()
                    mutexDraw.acquire()
                    for i in range(len(bbox)):
                        if(label[i] == 'car'):
                            self.objectbox.append(BoundingBoxes(bbox[i][0], bbox[i][1], (bbox[i][2] - bbox[i][0]), (bbox[i][3] - bbox[i][1]),'person'))
                    mutexDraw.release()
                    time.sleep(0.3)
            print("done waiting :")
        finally:
            if(mutexDetection.locked() == True):
                mutexDetection.release()
            if(mutexDraw.locked() == True):
                mutexDraw.release()

class mywebcam:
    def __init__(self, cameraNumber):
        self.cameraNumber = cameraNumber
        self.bIsAppExitCalled = False;
        self.objectbox = []
        self.nZeroFrameCount = 0

        if(g_bUseWebcam == True):
            self.webcam = cv2.VideoCapture(self.cameraNumber)
            self.webcam.open(self.cameraNumber)
            if not self.webcam.isOpened():
                raise Exception("Could not open video device")
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.frame = self.webcam.read()
        else:
            if(self.cameraNumber == 0):
                self.frame = cv2.imread("resized.png")
            else:
                self.frame = cv2.imread("c.jpeg")

        self.objectDet = ObjectDetection(self.frame)
        self.e = Event()
        self.t1 = Thread(target=self.objectDet.infer,args=(self.e,self.cameraNumber))
        self.t1.start()
    
    def StartThread(self):
        while self.bIsAppExitCalled == False:
            try:
                if(g_bUseWebcam == True):
                    status, image = self.webcam.read()
                else:
                    if(self.cameraNumber == 0):
                        image = cv2.imread("resized.png")
                    else:
                        image = cv2.imread("c.jpeg")
                self.frame = image
                
                self.objectDet.frame = self.frame
                
                self.e.set()
                self.DrawOnImage()

                cv2.imshow("Image"+str(self.cameraNumber), self.frame)
                if(cv2.waitKey(30) == ord('q')):
                    break
            except:
                pass
                
            finally:
                pass
        self.objectDet.bIsAppExitCalled = True
        cv2.destroyAllWindows()
        if(g_bUseWebcam == True):
            self.webcam.release()
        
    def DrawOnImage(self):
        try:
            mutexDraw.acquire()
            self.objectbox = self.objectDet.objectbox
            mutexDraw.release()

            nNumOfObjects = len(self.objectbox)
            if((nNumOfObjects == 0) & (self.nZeroFrameCount < 4)):
                self.nZeroFrameCount = self.nZeroFrameCount + 1
                return
            else:
                self.nZeroFrameCount = 0

            for i in range(nNumOfObjects):
                cv2.rectangle(self.frame, self.objectbox[i].rect(), (255, 0, 0))

            self.frame = cv2.putText(self.frame, 'cars :' + str(len(self.objectbox)),org, font, fontScale, color, thickness, cv2.LINE_AA)
        finally:
            if(mutexDraw.locked() == True):
                mutexDraw.release()

if __name__ == "__main__":
   
    Stream1 = mywebcam(0)
    thread1 = Thread(target=Stream1.StartThread, args=())

    Stream2 = mywebcam(1)
    thread2 = Thread(target=Stream2.StartThread, args=())
    
    thread1.start()
    thread2.start()
    
    thread2.join()

    
