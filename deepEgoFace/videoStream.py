import os
import numpy
import cv2


class videoStream( object):
    """
    Abstract class defining what is a video stream
    """
    def nextFrame(self):
        pass
    def isClosed(self):
        pass
    def getName(self):
        pass


class cameraStream():
    """
    video stream from a camera in realtime
    (int)device is the device number
    """
    def __init__( self, device):
        self.name = "camera"
        self.capture = cv2.VideoCapture(device)
        self.nbframes = 1
        if self.capture.isOpened():
            self.hasNext = True;
        else:
            self.hasNext = False
            raise Exception('ErrorVideoNotOppened')

    def __del__(self):
        self.capture.release()

    def nextFrame(self):
        self.hasNext, frame = self.capture.read()
        self.nbframes=self.nbframes+1
        return self.nbframes, frame

    def isClosed(self):
        return not self.hasNext

    def getName(self):
        return self.name





class opencvVideo( videoStream):
    """
    video stream from a video
    (str)filePath is the path to the video
    framePerSecondToExtract is the number of frame to extract per second
    """
    def __init__( self, filePath, framePerSecondToExtract=numpy.inf):
        self.name = os.path.basename(filePath).split(".", 1)[0]
        self.capture = cv2.VideoCapture(filePath)
        self.nbFramesSkip = int(self.capture.get(cv2.CAP_PROP_FPS)) / framePerSecondToExtract
        self.nbframes = 1
        if self.capture.isOpened():
            self.hasNext = True;
        else:
            self.hasNext = False
            raise Exception('ErrorVideoNotOppened')

    def __del__(self):
        self.capture.release()

    def nextFrame(self):
        self.hasNext, frame = self.capture.read()
        self.nbframes=self.nbframes+1
        #go just before the future frame to
        nbf = 1
        #optimize by tmpstring and then write
        while(self.hasNext and nbf < self.nbFramesSkip):
            self.hasNext, f = self.capture.read()
            nbf=nbf+1
            self.nbframes=self.nbframes+1
        return self.nbframes-self.nbFramesSkip, frame

    def isClosed(self):
        return not self.hasNext

    def getName(self):
        return self.name




class folderAsStream( videoStream):
    """
    Treat a folder as a stream
    (str)folderPath is the path to the folder
    (str)ext is the extension of the images
    """
    def __init__( self, folderPath, ext):
        self.name = folderPath
        self.imagesName = []
        files = os.listdir(folderPath)
        for name in files:
            if len(name.split(".", 1))>1 and name.split(".", 1)[1] == ext:
                self.imagesName.append(name)
        self.current = 0

    def nextFrame(self):
        frame =  cv2.imread(self.name+"/"+self.imagesName[self.current])
        self.current = self.current + 1
        return self.current, frame

    def isClosed(self):
        if self.current < len(self.imagesName):
            return False
        else:
            return True

    def getName(self):
        return self.name
