import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


TRAIN_DIR = 'train'
TEST_DIR = 'test'
VALIDATION_DIR = 'validatie'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'dogs-vs-cats-convnet'

def create_label(image_name):
    """ Create an one-hot encoded vector from image name """
    word_label = image_name.split('.')[-3]
    if word_label == 'cat':
        return np.array([1,0])
    elif word_label == 'dog':
        return np.array([0,1])

# Functies voor het maken van training, test en validatie sets
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data
def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img_data), create_label(img)])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data
def create_validation_data():
    validation_data = []
    for img in tqdm(os.listdir(VALIDATION_DIR)):
        path = os.path.join(VALIDATION_DIR,img)
        img_label = img.split('.')[-3]
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        validation_data.append([np.array(img_data), img_label])
    shuffle(validation_data)
    np.save('validation_data.npy', validation_data)
    return validation_data

# Maken van de training, test en validatie set
try:
    if os.path.isfile('train_data.npy') and os.path.isfile('test_data.npy') and os.path.isfile('validation_data.npy'):
        train_data = np.load('train_data.npy')
        test_data = np.load('test_data.npy')
        validation_data = np.load('validation_data.npy')
    else:
        train_data = create_train_data()
        test_data = create_test_data()
        validation_data = create_validation_data()
except:
        print("Unexpected error: ", sys.exc_info()[0])


X_train = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [i[1] for i in train_data]
X_test = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test_data]

# Tensorflow model
tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
          validation_set=({'input': X_test}, {'targets': y_test}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

# Validatie van het model adhv validatie data
correct = 0
for num, data in enumerate(validation_data):
    img_label = data[1]
    img_data = data[0]
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label='Dog'
    else:
        str_label='Cat'

    print(img_label,"\t",str_label)
    if str_label.lower() == img_label:
        correct += 1
perc_correct = (correct/len(validation_data))*100
print("percentage correct: ", perc_correct)
