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

grayscale = True
(Ximg, Yimg) = (64,64)
categories = ['jeremy', 'julien', 'nicolas', 'warren']
train_datagen = ImageDataGenerator(rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.4,
        rotation_range = 45,
        )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('./trainingSet',
                target_size = (Ximg, Yimg),
                batch_size = 32,
                class_mode = 'categorical',
                color_mode = 'grayscale' if grayscale else 'rgb')

test_set = test_datagen.flow_from_directory('./testSet',
                target_size = (Ximg,Yimg),
                batch_size = 32,
                class_mode = 'categorical',
                color_mode = 'grayscale' if grayscale else 'rgb')

# (x,y) = training_set[0] # Shuffled at each launch

# for i in range(0,4):
#     image = x[i]
#     if grayscale: image = np.reshape(image, (Ximg, Yimg))
#     print("Dimension of input image=",np.shape(image))
    
#     #plt.imshow(image.transpose(2,1,0))
#     if i>0: plt.figure()
#     if grayscale:
#         plt.imshow(image, cmap='gray')
#     else:
#         plt.imshow(image)
#     plt.show()
#     print(categories[int(y[i])])

colorChannels = 1 if grayscale else 3

model = Sequential()
model.add(Conv2D(filters = 32,kernel_size = 3,strides = 1,padding = 'same',activation = 'relu', input_shape=(Ximg, Yimg, colorChannels), data_format='channels_last'))
model.add(Conv2D(filters = 32,kernel_size = 3,strides = 1,padding = 'same',activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2, strides=None, padding='valid'))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 64,kernel_size = 3,strides = 1,padding = 'same',activation = 'relu'))
model.add(Conv2D(filters = 64,kernel_size = 3,strides = 1,padding = 'same',activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2, strides=None, padding='valid'))
model.add(Dropout(0.3))

model.add(Conv2D(filters = 128,kernel_size = 3,strides = 1,padding = 'same',activation = 'relu'))
model.add(Conv2D(filters = 128,kernel_size = 3,strides = 1,padding = 'same',activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2, strides=None, padding='valid'))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(4,kernel_initializer='uniform', activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#training
nb_sample_train = 10185
nb_sample_val = 2545
batch_size = 32
steps_train = int(nb_sample_train/batch_size)
steps_val = int(nb_sample_val/batch_size)

hist = model.fit_generator(generator=training_set, epochs=20, steps_per_epoch=steps_train,validation_steps=steps_val, verbose=1, validation_data=test_set)

print(model.summary())

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['train_loss','test_loss','train_acc','test_acc'],loc='upper_left')
plt.show()

model.save('./vgglike.h5')