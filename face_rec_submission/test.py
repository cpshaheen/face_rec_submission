import numpy as np 
import pandas as pd
import os, sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from PIL import Image

from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
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

# df = parse_dataset(dataset_folder_name)
# df.head()
# print(df)

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

def test_model(model, test_idx, data_generator):
    test_batch_size = 128
    test_generator = data_generator.generate_images(test_idx, is_training=False, batch_size=test_batch_size)
    race_pred, gender_pred = model.predict_generator(test_generator, 
                                                           steps=len(test_idx)//test_batch_size)
    test_generator = data_generator.generate_images(test_idx, 
        is_training=False, batch_size=test_batch_size)

    race_pred1 = race_pred
    samples = 0
    images, race_true, gender_true = [], [], []
    for test_batch in test_generator:
        image = test_batch[0]
        labels = test_batch[1]
        images.extend(image)
        race_true.extend(labels[0])
        gender_true.extend(labels[1])
        
    race_true1 = np.array(race_true)
    gender_true1 = np.array(gender_true)

    race_true = np.array(race_true)
    gender_true = np.array(gender_true)
    
    race_true, gender_true = race_true.argmax(axis=-1), gender_true.argmax(axis=-1)
    race_pred, gender_pred = race_pred.argmax(axis=-1), gender_pred.argmax(axis=-1)

    cr_race = classification_report(race_true, race_pred, 
        target_names=dataset_dict['race_alias'].keys())
    print(cr_race)


    cm = confusion_matrix(race_true, race_pred)
    print('Confusion Matrix')
    print('\t',end='')
    for i, race in enumerate(dataset_dict['race_id'].values()):
        t_r = dataset_dict['race_id'][i]
        if(i!=4):
            print('\t', end='')    
        else:
            print(' ', end='')
        print('P '+t_r, end='')
    print('\n')

    for i, race in enumerate(dataset_dict['race_id'].values()):
        t_r = dataset_dict['race_id'][i]
        
        print('A '+t_r, end='')
        cur_cm = cm[i]
        if(i<3):
            print('\t', end='')
        for val in cur_cm:
            print('\t'+str(val), end='')
        print('\n')

def main():
    # get all dirs of images
    dirs = [x for x in os.listdir() 
        if os.path.isdir(x) and x[0]!='.' and 'man' in x]
    df = parse_dataset2(dirs)
    df.head()
    if '-p' in sys.argv:
        plot_distribution(df['race'])
        plot_distribution(df['gender'])
    
    data_generator = FaceDataGenerator(df)
    train_idx, valid_idx, test_idx = data_generator.generate_split_indexes()
    model = load_model(sys.argv[1])
    test_model(model, test_idx, data_generator)
    

main()
