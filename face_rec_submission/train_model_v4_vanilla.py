import numpy as np 
import pandas as pd
import os, sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go

from PIL import Image
from datetime import date
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam

import tensorflow as tf

TYPE = 'vanilla'

today = date.today()

vgnum = 0

TRAIN_TEST_SPLIT = 0.7
IM_WIDTH = IM_HEIGHT = 200
dataset_dict = {
    'race_id': {
        0: 'white', 
        1: 'black', 
        2: 'asian', 
        3: 'indian', 
        4: 'latino'
    },
    'gender_id': {
        0: 'male',
        1: 'female'
    }
}

dataset_dict['gender_alias'] = dict((g, i) for i, g in dataset_dict['gender_id'].items())
dataset_dict['race_alias'] = dict((r, i) for i, r in dataset_dict['race_id'].items())

print(dataset_dict)

batch_sizes=[16,32,64]



def get_data(dir):
    # encode one hot race array
    for val in dataset_dict['race_id'].values():
        if val in dir:
            race = val

    # encode one hot gender array
    if 'woman' in dir:
        gender = 'female'
    else:
        gender = 'male'

    return gender, race

def parse_dataset2(dirs):

    records = []

    # iterate over all dirs and images in dirs and return a DF w/gender,sex
    for dir in dirs:
        gender, race = get_data(str(dir))
        for image in os.listdir(dir):
            records.append((gender,race,dir+'/'+str(image)))
    
    df = pd.DataFrame(records)
    df.columns = ['gender', 'race', 'image']
    df.dropna()

    return df


def plot_distribution(pd_series):
    labels = pd_series.value_counts().index.tolist()
    counts = pd_series.value_counts().values.tolist()
    
    pie_plot = go.Pie(labels=labels, values=counts, hole=.3)
    fig = go.Figure(data=[pie_plot])
    fig.update_layout(title_text='Distribution for %s' % pd_series.name)
    
    fig.show()

class FaceDataGenerator():
    """
    Custom Data Generator
    """
    def __init__(self, df):
        self.df = df
        
    def generate_split_indexes(self):
        p = np.random.permutation(len(self.df))
        train_up_to = int(len(self.df) * TRAIN_TEST_SPLIT)
        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]

        train_up_to = int(train_up_to * TRAIN_TEST_SPLIT)
        train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]
        
        # converts alias to id
        self.df['gender_id'] = self.df['gender'].map(lambda gender: dataset_dict['gender_alias'][gender])
        self.df['race_id'] = self.df['race'].map(lambda race: dataset_dict['race_alias'][race])
        
        return train_idx, valid_idx, test_idx
    
    def preprocess_image(self, img_path):
        """
        Used to perform some minor preprocessing on the image before inputting into the network.
        """
        im = Image.open(img_path)
        im = im.resize((IM_WIDTH, IM_HEIGHT))
        im = np.array(im) / 255.0
        
        return im
        
    def generate_images(self, image_idx, is_training, batch_size=16):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        """
        
        # arrays to store our batched data
        images, races, genders = [], [], []
        while True:
            for idx in image_idx:
                person = self.df.iloc[idx]
                
                race = person['race_id']
                gender = person['gender_id']
                file = person['image']
                
                im = self.preprocess_image(file)
                
                races.append(to_categorical(race, len(dataset_dict['race_id'])))
                genders.append(to_categorical(gender, len(dataset_dict['gender_id'])))
                images.append(im)
                
                # yielding condition
                if len(images) >= batch_size:
                    yield np.array(images), [np.array(races), np.array(genders)]
                    images, races, genders = [], [], []
                    
            if not is_training:
                break

class FaceDualOutputModel():

    def make_default_hidden_layers(self, inputs):

        # load VGG16
        vg_m = VGG16(include_top=False,
            weights=None)

        return vg_m

    def build_race_branch(self, inputs, num_races, vg_m):
        """
        Used to build the race branch of our face recognition network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks, 
        followed by the Dense output layer.
        """

        x = vg_m(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_races)(x)
        x = Activation("softmax", name="race_output")(x)

        return x

    def build_gender_branch(self, inputs, vg_m,num_genders=2):
        """
        Used to build the gender branch of our face recognition network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks, 
        followed by the Dense output layer.
        """
        # x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(inputs)

        x = vg_m(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_genders)(x)
        x = Activation("sigmoid", name="gender_output")(x)

        return x

    def assemble_full_model(self, width, height, num_races):
        """
        Used to assemble our multi-output model CNN.
        """
        input_shape = (height, width, 3)

        inputs = Input(shape=input_shape)

        vg_m = VGG16(include_top=False,
                weights=None)

        race_branch = self.build_race_branch(inputs, num_races, vg_m)
        gender_branch = self.build_gender_branch(inputs, vg_m)

        model = Model(inputs=inputs,
                     outputs = [race_branch, gender_branch],
                     name="face_net")

        return model

def train(model,data_generator,train_idx,valid_idx,b_size=32):
    init_lr = 1e-4
    epochs = 100

    opt = Adam(lr=init_lr, decay=init_lr / epochs)

    model.compile(optimizer=opt, 
              loss={
                  'race_output': 'categorical_crossentropy', 
                  'gender_output': 'binary_crossentropy'},
              loss_weights={
                  'race_output': 1.5, 
                  'gender_output': 0.1},
              metrics={
                  'race_output': 'accuracy',
                  'gender_output': 'accuracy'})

    batch_size = b_size
    valid_batch_size = b_size
    train_gen = data_generator.generate_images(train_idx, is_training=True, batch_size=batch_size)
    valid_gen = data_generator.generate_images(valid_idx, is_training=True, batch_size=valid_batch_size)

    callbacks = [
        ModelCheckpoint("./model_checkpoint", monitor='val_loss')
    ]

    # class weights to deal with imbalance of different races
    class_weights = {'race_output':{0:1,1:1.66,2:2,3:5.33,4:5.71},
        'gender_output': {0:1,1:1}}
    
    history = model.fit_generator(train_gen,
        steps_per_epoch=len(train_idx)//batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=valid_gen,
        validation_steps=len(train_idx)//batch_size,
        verbose=1)

    model.save('face_model_v4_'+TYPE+'_'+today.strftime("%m-%d-%y")+'.h5')

    return history

def acc_check(history, b_size):
    plt.clf()
    print(history.history)
   
    plt.plot(history.history['race_output_accuracy'])
    plt.plot(history.history['val_race_output_accuracy'])
    plt.title('race_accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper left')

    plt.savefig('acc_race_v4'+TYPE+'_'+today.strftime("%m-%d-%y")+'.png')
    
    plt.clf()

    plt.plot(history.history['gender_output_accuracy'])
    plt.plot(history.history['val_gender_output_accuracy'])
    plt.title('gender_accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper left')

    plt.savefig('acc_gender_v4'+TYPE+'_'+today.strftime("%m-%d-%y")+'.png')

def loss_check(history, b_size):
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('overall loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper left')
    
    plt.savefig('overall_loss_v4'+TYPE+'_'+today.strftime("%m-%d-%y")+'.png')

def main():
    # get all dirs of images
    dirs = [x for x in os.listdir() if os.path.isdir(x) and x[0]!='.' and 'man' in x]
    print(dirs)
    df = parse_dataset2(dirs)
    df.head()
    print(df)
    if '-p' in sys.argv:
        plot_distribution(df['race'])
        plot_distribution(df['gender'])
    batch_size = 32 
    data_generator = FaceDataGenerator(df)
    train_idx, valid_idx, test_idx = data_generator.generate_split_indexes()
    model = FaceDualOutputModel().assemble_full_model(IM_WIDTH, IM_HEIGHT, 
        num_races=len(dataset_dict['race_alias']))
    history = train(model,data_generator,train_idx,valid_idx, batch_size)
    acc_check(history, batch_size)
    loss_check(history, batch_size)

main()
