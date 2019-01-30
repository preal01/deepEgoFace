from sys import getsizeof
import os

from deepEgoFace import *

from sklearn import model_selection
from sklearn.cluster import KMeans
from sklearn import svm
import pickle








class SetExtractor:

    @staticmethod
    def extract(data,test_size):
        x = []
        y = []
        for elems in data:
            elemsL = elems.split(",")
            x.append([float(el) for el in elemsL[3].split(" ")])
            y.append(int(elemsL[4]))
        #print(getsizeof(x))
        xt, xtt, yt, ytt = model_selection.train_test_split(x, y, test_size=test_size, random_state=7)
        return [ xt, yt, xtt, ytt]
    #extract = staticmethod(extract)



class classifier( object):
    """
    Abstract class defining what is an classifer
    """
    def classify(self,embedding):
        pass
    def train(self, embbedingsClassesPath):
        pass






class kmeansBased:
    """
    not done yet
    """
    def __init__(self, modelPath, LabelsConversionPath):
        self.modelPath = modelPath
        self.justTrained = False
        if os.path.isfile(self.modelPath):
            self.model = pickle.load(open(self.modelPath, 'rb'))
        self.lPath = LabelsConversionPath
        if os.path.isfile(self.modelPath):
            self.setLabelsConversion(LabelsConversionPath)


    def statiticalyDeduceClass(self,k):
        self.labelConversion = [0 for i in range(0,k)]
        for l in range(0,k):
            max = 0
            for c in range(0,int(k/2)):
                nb = 0
                for y in range(0,len(self.trainY)):
                    if self.trainY[y] == c and self.model.labels_[y] == l:
                        nb = nb + 1
                if nb > max :
                    max = nb
                    self.labelConversion[l] = c
        file = open(self.lPath,"w+")
        file.write(str(self.labelConversion[0]))
        for l in range(1,len(self.labelConversion)):
            file.write(","+str(self.labelConversion[l]))


    def setLabelsConversion( self, LabelsConversionPath):
        file = open(LabelsConversionPath,"r")
        for line in file:
            self.labelConversion = [int(el) for el in line.split(',')]

    def evaluate(self):
        #silouhette_score ?
        trainL = self.model.labels_
        nbCorrect = 0
        nbTotal = 0
        for l in trainL:
            if self.labelConversion[l] == self.trainY[nbTotal]:
                nbCorrect = nbCorrect + 1
            nbTotal = nbTotal + 1
        trainAcc = float(nbCorrect)/nbTotal

        testL = self.model.predict(self.testX)
        nbCorrect = 0
        nbTotal = 0
        for l in testL:
            if self.labelConversion[l] == self.testY[nbTotal]:
                nbCorrect = nbCorrect + 1
            nbTotal = nbTotal + 1
        testAcc = float(nbCorrect)/nbTotal

        print("training accuray: "+ str(trainAcc) +"   validate accuray: "+ str(testAcc))



    def train( self, embbedingsClassesPath, k):
        file = open(embbedingsClassesPath,"r")
        data = []
        for line in file:
            data.append(line)
        file.close()
        [self.trainX, self.trainY, self.testX, self.testY] = SetExtractor.extract(data,0.4)
        self.model = KMeans(n_clusters=k, init='k-means++').fit(self.trainX)
        self.statiticalyDeduceClass(k)
        self.evaluate()
        pickle.dump(self.model, open(self.modelPath, 'wb'))

    def classify( self, embedding):
        return self.labelConversion[self.model.predict([embedding])[0]]






class svmBased(classifier):

    def __init__(self, modelPath):
        self.modelPath = modelPath
        if os.path.isfile(self.modelPath):
            self.model = pickle.load(open(self.modelPath, 'rb'))

    def train(self, embbedingsClassesPath):
        file = open(embbedingsClassesPath,"r")
        data = []
        for line in file:
            data.append(line)
        file.close()
        [trainX, trainY, TestX, TestY] = SetExtractor.extract(data,0.4)
        self.model = svm.SVC(gamma='scale', decision_function_shape='ovo')
        self.model.fit(trainX, trainY)
        #evaluate fit
        result = self.model.score(TestX, TestY)
        print(result)
        pickle.dump(self.model, open(self.modelPath, 'wb'))

    def classify(self, embedding):
        return self.model.predict([embedding])[0]
