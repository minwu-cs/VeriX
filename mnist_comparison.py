import os
import time
import random
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr
from keras.datasets import mnist
from keras.models import load_model
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from maraboupy import Marabou

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=10)
parser.add_argument('--seed', type=int, default=137)
parser.add_argument('--epsilon', type=float, default=0.1)
args = parser.parse_args()

# Load test data
random.seed(args.seed)
(_, _), (x_test, y_test) = mnist.load_data()
num_test = x_test.shape[0]
x_test = x_test.reshape(num_test, 28, 28, 1)
y_test = tf.keras.utils.to_categorical(y_test, 10)
x_test = x_test.astype('float32') / 25
indices = random.sample(range(0, num_test), args.num_samples)
x = x_test[indices]
y = y_test[indices]

# Load pre-trained networks

# keras_model_path = directory + model_name + '.h5'
# keras_model = load_model(keras_model_path)
# keras_model.summary()
# score = keras_model.evaluate(x, y, verbose=0)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])

networks_path = 'networks/'
h5_models = []
for file in os.listdir(networks_path):
    if file.startswith('mnist'):
        if file.endswith('.h5'):
            keras_model_path = networks_path + file
            keras_model = load_model(keras_model_path)
            keras_model.summary()
            score = keras_model.evaluate(x_test, y_test, verbose=0)
            h5_models.append(keras_model)
            print("Test loss:", score[0])
            print("Test accuracy:", score[1])
            print("Loaded model", keras_model.name)
            print('*'*60, '\n')

# For each network, produce explanations on the set of images

# Visualization