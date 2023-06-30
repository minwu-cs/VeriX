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
from utils import suppress_stdout, plot_figure
from verix import VeriX


TIMEOUT = 60
directory = 'models/'
if not os.path.exists(directory):
    os.mkdir(directory)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--network', type=str, default='mnist-10x2')
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--verbosity', type=bool, default=False)
args = parser.parse_args()

dataset = args.dataset
model_name = args.network
index = args.index
epsilon = args.epsilon

result_dir = 'outputs/index-%d-%s-%ds-heuristic-linf%g-test' % (index, model_name, TIMEOUT, epsilon)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x = x_test
y = y_test

keras_model_path = directory + model_name + '.h5'
solver = VeriX(keras_model_path, x[index], y[index])
solver.add_traversal_order('sensitivity_reversal')
sat_set, unsat_set, timeout_set = solver.generate_explanation('sensitivity_reversal', epsilon, verbosity=args.verbosity)

image = x[solver.pred].flatten()
mask = np.zeros(image.shape).astype(bool)
mask[sat_set] = True
mask[timeout_set] = True

plot_figure(image=label2rgb(mask.reshape(28, 28), x[index].reshape(28, 28),
                            # colors=[[1, 1, 0]],
                            # colors=[[128 / 255, 1, 0]],
                            colors=[[0, 1, 0]],
                            bg_label=0),
            path='%s/index-%d-%s-linf%g-explanation-%d.png' %
                 (result_dir, index, model_name, epsilon, len(sat_set)+len(timeout_set)))

mask = np.zeros(image.shape).astype(bool)
mask[timeout_set] = True

plot_figure(image=label2rgb(mask.reshape(28, 28), x[index].reshape(28, 28),
                            # colors=[[1, 1, 0]],
                            # colors=[[128 / 255, 1, 0]],
                            colors=[[0, 1, 0]],
                            bg_label=0),
            path='%s/index-%d-%s-linf%g-timeout-%d.png' %
                 (result_dir, index, model_name, epsilon, len(timeout_set)))

np.savetxt('%s/index-%d-%s-linf%g-unsat.txt' % (result_dir, index, model_name, epsilon),
           unsat_set, fmt='%d')
np.savetxt('%s/index-%d-%s-linf%g-sat.txt' % (result_dir, index, model_name, epsilon),
           sat_set, fmt='%d')
np.savetxt('%s/index-%d-%s-linf%g-timeout.txt' % (result_dir, index, model_name, epsilon),
           timeout_set, fmt='%d')
