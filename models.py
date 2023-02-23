import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,Model
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

def VGG16(img_size=224,num_classes=17):
    model = Sequential([
        Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', input_shape=(img_size, img_size, 3),activation='relu'),
        Conv2D(64, (3,3), (1, 1), 'same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        Conv2D(128, (3,3), (1, 1), 'same', activation='relu'),
        Conv2D(128, (3,3), (1, 1), 'same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        Conv2D(256, (3,3), (1, 1), 'same', activation='relu'),
        Conv2D(256, (3,3), (1, 1), 'same', activation='relu'),
        Conv2D(256, (3,3), (1, 1), 'same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        Conv2D(512, (3,3), (1, 1), 'same', activation='relu'),
        Conv2D(512, (3,3), (1, 1), 'same', activation='relu'),
        Conv2D(512, (3,3), (1, 1), 'same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        Conv2D(512, (3, 3), (1, 1), 'same', activation='relu'),
        Conv2D(512, (3, 3), (1, 1), 'same', activation='relu'),
        Conv2D(512, (3, 3), (1, 1), 'same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        Flatten(),
        Dense(4096, 'relu'),
        Dropout(0.5),
        Dense(4096, 'relu'),
        Dropout(0.5),
        Dense(num_classes, 'softmax')
    ])

    return model

def Inception(x,filters):

    x_1 = Conv2D(filters[0],1,1,'same',activation='relu')(x)

    x_2 = Conv2D(filters[1],1,1,'same',activation='relu')(x)
    x_2 = Conv2D(filters[2],3,1,'same',activation='relu')(x_2)

    x_3 = Conv2D(filters[3], 1, 1, 'same', activation='relu')(x)
    x_3 = Conv2D(filters[4], 5, 1, 'same', activation='relu')(x_3)

    mp = MaxPooling2D(3,1,'same')(x)
    mp = Conv2D(filters[5],1,1,'same',activation='relu')(mp)
    x = Concatenate(axis=3)([x_1,x_2,x_3,mp])

    return x

def GoogleNet(img_size=224,num_classes=17):
    model_input = Input(shape=(img_size,img_size,3))

    x = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same',activation='relu')(model_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2D(64, (1,1), (1, 1), 'same', activation='relu')(x)
    x = Conv2D(192, (3,3), (1, 1), 'same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Inception(x,[64,96,128,16,32,32])
    x = Inception(x,[128,128,192,32,96,64])
    x = MaxPooling2D(3,2,'same')(x)
    x = Inception(x, [196,86,208,16,48,64])
    x = Inception(x, [160,112,224,24,64,64])
    x = Inception(x, [128,128,256,24,64,64])
    x = Inception(x, [112,144,288,32,64,64])
    x = Inception(x, [256,160,320,32,128,128])
    x = MaxPooling2D(3, 2, 'same')(x)
    x = Inception(x, [256,160,320,32,128,128])
    x = Inception(x, [384,192,384,48,128,128])

    x = AveragePooling2D(7,7,'same')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, 'softmax')(x)

    model = Model(model_input,x)

    return model



if __name__ == '__main__':
    GoogleNet().summary()