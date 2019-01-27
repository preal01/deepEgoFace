from sys import getsizeof
import os

from sklearn import model_selection
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



class kmeans:
    """
    not done yet
    """
    def __init__(self, modelPath):
        x = []





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
