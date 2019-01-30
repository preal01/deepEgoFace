

import sys

from deepEgoFace import *
from deepEgoFace.recognizer import *



det = detector.opencvBased( "./model/detector/opencv/deploy.prototxt.txt","./model/detector/opencv/res10_300x300_ssd_iter_140000.caffemodel")

rec = faceClassifier.kerasCnn("./model/recognizer/face-classifier/corpusAndOther_res18.h5")

labels = utils.getLabels("corpus/faces")
labels = sorted(labels)

stream = videoStream.cameraStream(int(sys.argv[1]))
fs = workflow.faceStream( stream,
                         "faces-tmp/",
                         "labels.csv",
                         det,
                         rec,
                         display=True,
                         labels=labels) #,
                         #align="model/recognizer/align/shape_predictor_68_face_landmarks.dat")
fs.recognizeFromAll()
