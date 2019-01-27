import cv2


class embedder( object):
    """
    Abstract class defining what is an embedder
    """
    def embed(self,face):
        pass



class openface(embedder):

    def __init__(self, modelPath):
        self.net = cv2.dnn.readNetFromTorch(modelPath)

    def embed(self,face):
        #face = self.align.align( 96, frame, bb=roi)
        faceBlob = cv2.dnn.blobFromImage( face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(faceBlob)
        vec = self.net.forward()
        return vec.flatten()
