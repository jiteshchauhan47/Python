import cvlib as cv
from cvlib.object_detection import draw_bbox
import sys
import cv2
from threading import Thread, Lock

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (0, 255, 0)
thickness = 2
org = (50, 50)

mutexDetection = Lock()
mutexDraw = Lock()
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

class mywebcam:
    def __init__(self, cameraNumber):
        self.cameraNumber = cameraNumber
        self.webcam = cv2.VideoCapture(self.cameraNumber)
        self.webcam.open(self.cameraNumber)
        self.bIsAppExitCalled = False;
        self.objectbox = []
        self.nZeroFrameCount = 0
        if not self.webcam.isOpened():
            raise Exception("Could not open video device")
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def infer(self,image):
        try:
            mutexDetection.acquire()
            bbox, label, conf = cv.detect_common_objects(image, 0.5, 0.3, 'yolov4-tiny', False)
            self.objectbox.clear()
            for i in range(len(bbox)):
                if(label[i] == 'person'):
                    self.objectbox.append(BoundingBoxes(bbox[i][0], bbox[i][1], (bbox[i][2] - bbox[i][0]), (bbox[i][3] - bbox[i][1]),'person'))
        finally:
            mutexDetection.release()

    def StartThread(self):
        while self.bIsAppExitCalled == False:
            try:
                status, image = self.webcam.read()
                self.frame = image
                self.infer(image)
                self.DrawOnImage()

                cv2.imshow("Image"+str(self.cameraNumber), self.frame)
                if(cv2.waitKey(10) == ord('q')):
                    break
            finally:
                pass
        self.webcam.release()
        cv2.destroyAllWindows()
    
    def DrawOnImage(self):
        nNumOfObjects = len(self.objectbox)
        if((nNumOfObjects == 0) & (self.nZeroFrameCount < 4)):
            self.nZeroFrameCount = self.nZeroFrameCount + 1
            return
        else:
            self.nZeroFrameCount = 0
            
        for i in range(nNumOfObjects):
                cv2.rectangle(self.frame, self.objectbox[i].rect(), (255, 0, 0))

        self.frame = cv2.putText(self.frame, 'cars :' + str(len(self.objectbox)),org, font, fontScale, color, thickness, cv2.LINE_AA)

if __name__ == "__main__":
   
    Stream1 = mywebcam(0)
    thread1 = Thread(target=Stream1.StartThread, args=())
    thread1.start()
    thread1.join()
    
