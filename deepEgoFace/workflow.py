from . import videoStream
from . import align_dlib
import cv2
import dlib

import os
import errno
import time

import re
import numpy as np


#save parameters for the face recognizer streamer
SAVE_NOTHING                 = 0
SAVE_DETECTION_FRAMEBBS      = 1
SAVE_DETECTION_CROPS         = 2
SAVE_RECOGNIZED_FRAMEBBS     = 4
SAVE_RECOGNIZED_CLASSES      = 8
SAVE_RECOGNIZED_EMBEDDINGS   = 16






class faceStream:
    """
    class used to associate a workflow to a stream of image
    """
    def __init__(self,
                 videoStream,
                 faceFolderPath,
                 classFilePath,
                 dectector,
                 recognizer,
                 save=SAVE_NOTHING,
                 display=False,
                 labels=None,
                 align=None
                ):

        self.stream = videoStream
        classFolderPath = os.path.dirname(classFilePath)
        try:
            os.makedirs(faceFolderPath)
            if classFolderPath is not '':
                os.makedirs(classFolderPath)
        except OSError as exc:
            pass
        self.detectionFolder = faceFolderPath
        self.classFile = open(classFilePath,"a+")
        self.detector = dectector
        self.recognizer = recognizer
        self.save = save
        self.display = display
        self.labels = labels
        if align is not None:
            self.align = align_dlib.AlignDlib(align)
        else:
            self.align = None


    def __del__(self):
        del self.stream
        cv2.destroyAllWindows()
        self.classFile.close()



    def detectFromAll( self, timeLogger=None):
        while(not self.stream.isClosed()):
            start = time.time()
            framenb, detections = self.detectFromNext(timeLogger=timeLogger)
            if timeLogger is not None:
                timeLogger.add( self.stream.getName(), framenb, "detectior-full",time.time()-start)


    def detectFromNext( self, timeLogger=None):

        framenb, frame = self.stream.nextFrame()
        if frame is None:
            return framenb, []
        start = time.time()
        detections = self.detector.detect(frame)
        if timeLogger is not None:
            timeLogger.add( self.stream.getName(), framenb, "detector-only",time.time()-start)

        #loop through detections
        nbdetect=1
        detectionInImage = frame.copy()
        for roi in detections :
            #add bbox to display
            if self.display or self.save & SAVE_DETECTION_FRAMEBBS:
                detectionInImage = cv2.rectangle(detectionInImage, (roi[0][0],roi[0][1]), (roi[1][0],roi[1][1]), (255,0,0), 2)
            #save crop
            if self.save & SAVE_DETECTION_CROPS:
                crop = frame[ roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
                (fH, fW) = crop.shape[:2]
                if fW > 20 and fH > 20:
                    cv2.imwrite(self.detectionFolder+self.stream.getName()+"-"+str(int(framenb))+"-"+str(nbdetect)+".png",crop)
                    nbdetect=nbdetect+1

        if self.display:
            cv2.imshow('Video',detectionInImage)
            cv2.waitKey(1)
        if self.save & SAVE_DETECTION_FRAMEBBS:
            cv2.imwrite(self.detectionFolder+self.stream.getName()+"-"+str(int(framenb))+"-detect.png",detectionInImage);
        return framenb, detections




    def recognizeFromAll( self, timeLogger=None):
        while(not self.stream.isClosed()):
            start = time.time()
            framenb, classes = self.recognizeFromNext(timeLogger=timeLogger)
            if timeLogger is not None:
                timeLogger.add( self.stream.getName(), framenb, "recognizer-full",time.time()-start)


    def recognizeFromNext( self, timeLogger=None):

        framenb, frame = self.stream.nextFrame()
        if frame is None:
            return framenb, []
        start = time.time()
        detections = self.detector.detect(frame)
        if timeLogger is not None:
            timeLogger.add( self.stream.getName(), framenb, "detector-only", time.time()-start)

        #loop through detections
        facenb=1
        classes=[]
        recognizedInImage = frame.copy()
        for roi in detections :
            if self.align is not None:
                bb = dlib.rectangle(left=roi[0][0], top=roi[0][1], right=roi[1][0], bottom=roi[1][1])
                face = self.align.align( 96, frame, bb=bb)
            else:
                face = frame[ roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]

            (fH, fW) = face.shape[:2]
            if fW > 20 and fH > 20:
                if self.save & SAVE_DETECTION_CROPS:
                    cv2.imwrite(self.detectionFolder+self.stream.getName()+"-"+str(int(framenb))+"-"+str(facenb)+".png",face)
                start = time.time()
                #recognize
                classes.append(self.recognizer.recognize(face))
                #log time
                if timeLogger is not None:
                    timeLogger.add( self.stream.getName(), framenb, facenb, "recognizer-only", time.time()-start)
                # add bounding box
                if self.display or self.save & SAVE_RECOGNIZED_FRAMEBBS:
                    recognizedInImage = cv2.rectangle(recognizedInImage, (roi[0][0],roi[0][1]), (roi[1][0],roi[1][1]), (255,0,0), 2)
                    if self.labels is not None:
                        cv2.putText(recognizedInImage,self.labels[classes[facenb-1][0]],(roi[0][0],roi[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
                if self.save & SAVE_RECOGNIZED_CLASSES+SAVE_RECOGNIZED_EMBEDDINGS:
                    self.classFile.write(self.stream.getName()+","+str(int(framenb))+","+str(facenb)+",")
                    if self.save & SAVE_RECOGNIZED_CLASSES and classes[facenb-1][0] is not None:
                        self.classFile.write(classes[facenb-1][0]+",")
                    if self.save & SAVE_RECOGNIZED_EMBEDDINGS:
                        self.classFile.write(' '.join(re.sub( '\n', '', str(classes[facenb-1][1]).strip('[  ]')).split())+",")
                    self.classFile.write("\n")
                facenb=facenb+1

        if self.display:
            cv2.imshow('Video',recognizedInImage)
            cv2.waitKey(1)
        if self.save & SAVE_DETECTION_FRAMEBBS:
            cv2.imwrite(self.detectionFolder+self.stream.getName()+"-"+str(int(framenb))+"-recog.png",recognizedInImage);
        return framenb, classes











class faceVideoCorpus:
    """
    class used to associate a workflow to a corpus of stream of image
    """
    def __init__(self, folderPath, ext, detector, recognizer, save=SAVE_NOTHING, align=None, timeLogger=None, framePerSecondToExtract=1):
        self.videosFolder = folderPath
        self.videosName = []
        files = os.listdir(folderPath)
        for name in files:
            if len(name.split(".", 1))>1 and name.split(".", 1)[1] == ext:
                self.videosName.append(name)
        self.current = 0
        self.detector = detector
        self.recognizer = recognizer
        self.save = save
        self.logger = timeLogger
        self.fpsExt = framePerSecondToExtract

    def detectInAll(self):
        for v in self.videosName:
            print(str(self.current+1) +"/"+str(len(self.videosName))+": "+v)
            self.detectInVideo(v)
        self.current = 0

    def detectInVideo(self,vname):
        stream = videoStream.opencvVideo( self.videosFolder+"/"+vname, framePerSecondToExtract=self.fpsExt)
        fs = faceStream(
            stream,
            "faces/",
            "./classes.csv",
            self.detector,
            self.recognizer,
            save=self.save,
            display=False)
        fs.detectFromAll(timeLogger=self.logger)
        del fs
        self.current=self.current+1

    def recognizeInAll(self):
        for v in self.videosName:
            print(str(self.current+1) +"/"+str(len(self.videosName))+": "+v)
            self.recognizeInVideo(v)
        self.current = 0

    def recognizeInVideo(self,vname):
        stream = videoStream.opencvVideo( self.videosFolder+"/"+vname, framePerSecondToExtract=self.fpsExt)
        fs = faceStream(
            stream,
            "faces/",
            "./classes.csv",
            self.detector,
            self.recognizer,
            save=self.save,
            display=False)
        fs.recognizeFromAll(timeLogger=self.logger)
        del fs
        self.current=self.current+1
