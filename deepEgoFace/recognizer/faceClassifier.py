import os
from shutil import copyfile

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.models import load_model

from . import resnet
import cv2
from sklearn import model_selection




class recognizer( object):
    """
    Abstract class defining what is a recognizer
    """
    def recognize(self,face):
        pass


class embedClassify(recognizer):

    def __init__(self,embedder,classif):
        self.embedder = embedder
        self.classifier = classif
        self.descriptor = [ "embeddings", "classes"]

    def recognize(self,face):
        embed = self.embedder.embed(face)
        if  self.classifier is not None:
            cls = self.classifier.classify(embed)
        else:
            cls = None
        return [ cls, embed]




class kerasCnn(recognizer):

    def __init__( self, modelPath):
        self.modelPath = modelPath
        if os.path.isfile(self.modelPath):
            self.model = load_model(modelPath)
        self.grayscale = True
        (self.Ximg, self.Yimg)  = (64,64)

    def recognize( self, face):
        if self.grayscale : img = cv2.cvtColor( face, cv2.COLOR_BGR2GRAY);
        img = cv2.resize(img, (self.Ximg, self.Yimg))
        #img = cv2.normalize(img, 0, 1, cv2.NORM_MINMAX)
        img = img * 1./255
        #if self.grayscale : img = np.reshape(img, (self.Ximg, self.Yimg))
        prediction = self.model.predict(np.array([np.reshape(img,(self.Ximg,self.Yimg,1))]))
        m = max(max(prediction))
        for c in range(0,len(prediction[0])):
            if prediction[0][c] == m:
                return [c]



class res18Based(kerasCnn):

    def train( self, labeledFaceFolder):
        #create model
        print("prepare data")
        batch_size = 256
        nb_categories, nb_sample_train, nb_sample_val = self.subdivideData( labeledFaceFolder, 0.33)
        colorChannels = 1 if self.grayscale  else 3
        print("create model")
        self.createModel( colorChannels, self.Ximg, self.Yimg, nb_categories)

        #prepare data
        print("data flow from directory")
        train_datagen = ImageDataGenerator(rescale = 1./255,
                shear_range = 0.2,
                zoom_range = 0.4,
                rotation_range = 45,
                )
        test_datagen = ImageDataGenerator(rescale = 1./255)
        training_set = train_datagen.flow_from_directory(labeledFaceFolder+'/../trainingSet',
                        target_size = (self.Ximg, self.Yimg),
                        batch_size = batch_size,
                        class_mode = 'categorical',
                        color_mode = 'grayscale' if self.grayscale  else 'rgb')
        test_set = test_datagen.flow_from_directory(labeledFaceFolder+'/../testSet',
                        target_size = (self.Ximg,self.Yimg),
                        batch_size = batch_size,
                        class_mode = 'categorical',
                        color_mode = 'grayscale' if self.grayscale  else 'rgb')

        #train
        steps_train = int(nb_sample_train/batch_size)
        steps_val = int(nb_sample_val/batch_size)
        print("train model")
        hist = self.model.fit_generator(generator=training_set, epochs=20, steps_per_epoch=steps_train,validation_steps=steps_val, verbose=1, validation_data=test_set)
        self.model.save(self.modelPath)

        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.plot(hist.history['acc'])
        plt.plot(hist.history['val_acc'])
        plt.legend(['train_loss','test_loss','train_acc','test_acc'],loc='upper_left')
        plt.show()


    def subdivideData( self, labeledFaceFolder, coef):
        #for onther just before this step rm all but nb data
        #find /path/to/dir -type f -print0 | sort -zR | tail -zn +1001 | xargs -0 rm
        x = []
        y = []
        nb_category = 0
        dirs = os.listdir(labeledFaceFolder)
        for dir in dirs:
            files = os.listdir(labeledFaceFolder+"/"+dir)
            for file in files:
                x.append(file)
                y.append(nb_category)
            nb_category = nb_category + 1

        xt, xtt, yt, ytt = model_selection.train_test_split(x, y, test_size=coef, random_state=7)
        if not os.path.isdir(labeledFaceFolder+"/../trainingSet/") and not os.path.isdir(labeledFaceFolder+"/../testSet/"):
            os.makedirs(labeledFaceFolder+"/../trainingSet/")
            os.makedirs(labeledFaceFolder+"/../testSet/")
            for dir in dirs:
                os.makedirs(labeledFaceFolder+"/../trainingSet/"+dir)
                os.makedirs(labeledFaceFolder+"/../testSet/"+dir)
            for x in range(0,len(xt)):
                copyfile( labeledFaceFolder+"/"+dirs[yt[x]]+"/"+xt[x], labeledFaceFolder+"/../trainingSet/"+dirs[yt[x]]+"/"+xt[x])
            for x in range(0,len(xtt)):
                copyfile( labeledFaceFolder+"/"+dirs[ytt[x]]+"/"+xtt[x], labeledFaceFolder+"/../testSet/"+dirs[ytt[x]]+"/"+xtt[x])

        return nb_category, len(xt), len(xtt)


    def createModel( self, colorChannels, Ximg, Yimg, categories):
        model = resnet.ResnetBuilder.build_resnet_18((colorChannels, Ximg, Yimg), categories)
        model.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
