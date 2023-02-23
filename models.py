import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

def AlexNet(img_size=224,num_classes=17):
    model = Sequential([
        Conv2D(filters=96,kernel_size=(11,11),strides=(4,4),padding='valid',input_shape=(img_size,img_size,3),activation='relu'),
        MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid'),
        Conv2D(256,(5,5),(1,1),'same',activation='relu'),
        MaxPooling2D((3,3),(2,2),'valid'),
        Conv2D(384,(3,3),(1,1),'same',activation='relu'),
        Conv2D(384,(3,3),(1,1),'same',activation='relu'),
        Conv2D(256,(3,3),(1,1),'same',activation='relu'),
        MaxPooling2D((3,3),(2,2),'valid'),
        Flatten(),
        Dense(4096,'relu'),
        Dropout(0.5),
        Dense(4096,'relu'),
        Dropout(0.5),
        Dense(num_classes,'softmax')
    ])

    return model

