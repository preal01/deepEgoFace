

import sys

from deepEgoFace import *
from deepEgoFace.recognizer import *



det = detector.opencvBased( "./model/detector/opencv/deploy.prototxt.txt","./model/detector/opencv/res10_300x300_ssd_iter_140000.caffemodel")

#emb = embedder.openface("./model/recognizer/embedder/openface_nn4.small2.v1.t7")
#cls = embeddingClassifier.svmBased("./model/recognizer/embedding-classifier/aligned_corpOnly_embedding_svm_model.sav")
#rec = faceClassifier.embedClassify( emb, cls)

rec = faceClassifier.res18Based()


labels = utils.getLabels("./corpus/faces")
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
