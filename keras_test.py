import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv
import math
import numpy as np

def loaddata():
    train_x = []
    train_y = []
    n_row = 0
    path = './train.csv'
    text = open(path, 'r')
    row = csv.reader(text)
    for r in row:
        temp = []
        if n_row != 0:
            train_y.append(r[0])
            temp = [float(b) for b in r[1].split(' ')]
            train_x.append(temp)
        n_row = n_row + 1

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    len = math.floor(train_x.shape[0]*0.8)
    x_train = train_x[0:len]
    y_train = train_y[0:len]
    x_test = train_x[len:]
    y_test = train_y[len:]

    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
    return (x_train, y_train), (x_test, y_test)

def Model():

    batch_size = 128
    num_classes = 7
    epochs = 100
    
    img_rows, img_cols = 48, 48
    input_shape = (img_rows, img_cols, 1)

    (x_train, y_train), (x_test, y_test) = loaddata()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation = 'relu', input_shape = input_shape))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])
    model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose = 0)

    print('Test Loss: ', score[0])
    print('Test accuracy: ', score[1])

if __name__ == '__main__':
    Model()
