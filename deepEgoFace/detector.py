import cv2
import dlib

import numpy as np


class detector( object):
    """
    Abstract class defining what is a detector
    """
    def detect(self,image):
        pass



class dlibBased(detector):

    def __init__(self, modelPath):
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1(modelPath)

    def detect(self,image):
        if image is None:
            return []
        #(h, w) = image.shape[:2]
        #image = cv2.resize(image, (300, 300))
        dlibDet = self.cnn_face_detector(image,0)
        detection = []
        for i,rec in enumerate(dlibDet):
            #detection.append([ [math.floor(rec.rect.left()*(w/300)),math.floor(rec.rect.top()*(h/300))] , [math.ceil(rec.rect.right()*(w/300)),math.ceil(rec.rect.bottom()*(h/300))] ])
            detection.append([ [rec.rect.left(),rec.rect.top()] , [rec.rect.right(),rec.rect.bottom()] ])
        return detection


class opencvBased(detector):

    def __init__(self, modelPath, weigthPath):
        self.net = cv2.dnn.readNetFromCaffe( modelPath, weigthPath)

    def detect(self,image):
        if image is None:
            return []
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        cvDet = self.net.forward()
        detection = []

        for i in range(0, cvDet.shape[2]):
            confidence = cvDet[0, 0, i, 2]
            if confidence > 0.6:
                box = cvDet[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                detection.append([ [startX,startY] , [endX,endY] ])
        return detection
