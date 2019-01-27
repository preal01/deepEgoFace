#https://github.com/raghakot/keras-resnet

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

import sys
sys.path.append('../keras-resnet')
import resnet


# Dimension of image 
grayscale = True
(Ximg, Yimg) = (64,64)

#Classes
categories = ['jeremy', 'julien', 'nicolas', 'warren']

#Data augmentation
train_datagen = ImageDataGenerator(rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.4,
        rotation_range = 45,
        )

test_datagen = ImageDataGenerator(rescale = 1./255)


#Reading data from directory
training_set = train_datagen.flow_from_directory('./trainingSet',
                target_size = (Ximg, Yimg),
                batch_size = 128,
                class_mode = 'categorical',
                color_mode = 'grayscale' if grayscale else 'rgb')

test_set = test_datagen.flow_from_directory('./testSet',
                target_size = (Ximg,Yimg),
                batch_size = 128,
                class_mode = 'categorical',
                color_mode = 'grayscale' if grayscale else 'rgb')

colorChannels = 1 if grayscale else 3

#Define the model
model = resnet.ResnetBuilder.build((colorChannels, Ximg, Yimg), len(categories), resnet.basic_block, [1, 1, 1, 1])
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


#training

#number of images in training and validation
nb_sample_train = 10185
nb_sample_val = 2545


batch_size = 128
steps_train = int(nb_sample_train/batch_size)
steps_val = int(nb_sample_val/batch_size)

hist = model.fit_generator(generator=training_set, epochs=20, steps_per_epoch=steps_train,validation_steps=steps_val, verbose=1, validation_data=test_set)

print(model.summary())

#Printing loss and accuracy
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['train_loss','test_loss','train_acc','test_acc'],loc='upper_left')
plt.show()


#Save the model
model.save('./test_res10.h5')

