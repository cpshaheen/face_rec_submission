import os
from keras import layers, Input
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from keras.layers import Flatten
from numpy import load
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from datetime import date

import sys


def build_model():
    # define input shape
    in_shp = (200,200,3)

    # load VGG16
    vg_m = VGG16(include_top=False,
                weights=None,
                input_shape=in_shp)
    # add new classifier layers
    flat1 = Flatten()(vg_m.layers[-1].output)
    class1 = Dense(128, activation='relu', 
                    kernel_initializer='he_uniform')(flat1)
    # build two seperate outputs for race and gender
    race_output = Dense(5, activation='softmax', name='race')(class1)
    gender_output = Dense(1, activation='sigmoid', name='gender')(class1)
    # define new model
    model = Model(inputs=vg_m.inputs, outputs=[race_output,gender_output])
    # specify loss functions for race and gender along with weighting both
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt,
                    loss = {'race': 'categorical_crossentropy',
                                    #'binary_crossentropy'
                            'gender': 'binary_crossentropy'},
                    loss_weights= {'race': 1.,
                                    'gender': 10.})

    model.summary()
    return model

# load train and test dataset
def load_dataset():
    # load dataset
    data = load(sys.argv[1])
    X, Y, Z = data['arr_0'], data['arr_1'], data['arr_2']
    # separate into train and test datasets
    trainX, testX, trainY, testY, trainZ, testZ = train_test_split(X, Y, Z, test_size=0.3, random_state=1)
    print(trainX.shape, trainY.shape, trainZ.shape, testX.shape, testY.shape, testZ.shape)
    return trainX, trainY, trainZ, testX, testY, testZ

def main():
    # load the dataset
    trainX, trainY, trainZ, testX, testY, testZ = load_dataset()

    # Normalize data
    trainX = trainX / 255
    testX = testX / 255
    
    # build/load the model
    model = build_model()
    
    # fit model
    history = model.fit(trainX, [trainY, trainZ], epochs=100, batch_size=64, verbose=1)
    print(history)

    # save the labeled images and the metadata file of the counts of categories
    today = date.today()

    # save model
    model.save('final_model_'+today.strftime("%m-%d-%y")+'.h5')
    
    # evaluate model
    results = model.evaluate(testX, [testY, testZ], batch_size=64)
    print(results)


main()
