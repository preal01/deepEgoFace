import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.models import load_model
from random import randint
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

#Loading the model
name = './test_res10.h5'
model = load_model(name)



# nb_sample_train = 10185
# nb_sample_val = 2545
# batch_size = 378
# steps_train = int(nb_sample_train/batch_size)
# steps_val = int(nb_sample_val/batch_size)

# print(model.evaluate_generator(test_set,steps = steps_val ))
