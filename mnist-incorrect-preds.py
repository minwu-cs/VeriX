import os
import argparse
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import load_model
from skimage.color import label2rgb
from utils import plot_figure
from verix import VeriX


directory = 'networks/'
if not os.path.exists(directory):
    os.mkdir(directory)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--network', type=str, default='mnist-10x2')
parser.add_argument('--epsilon', type=float, default=0.05)
parser.add_argument('--timeout', type=int, default=60)
parser.add_argument('--verbosity', type=bool, default=False)
args = parser.parse_args()

dataset = args.dataset
model_name = args.network
epsilon = args.epsilon
timeout = args.timeout

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
# x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

keras_model_path = directory + model_name + '.h5'
keras_model = load_model(keras_model_path)
logits = keras_model.predict(x_test)
preds = np.argmax(logits, axis=1)
correct_preds = np.argmax(y_test, axis=1)
incorrect_indices = np.where(preds != correct_preds)[0]
np.savetxt('outputs/incorrect-preds/%s-incorrect-indices.txt' % (model_name), incorrect_indices, fmt='%d')

for index in incorrect_indices[:100]:
    result_dir = 'outputs/incorrect-preds/index-%d-%s-%ds-heuristic-linf%g' % (index, model_name, timeout, epsilon)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    plot_figure(image=x[index],
                path='%s/index-%d-original-%d-predicted-as-%d.png' %
                     (result_dir, index, correct_preds[index], preds[index]),
                cmap='gray')
    np.savetxt('%s/index-%d-logits.txt' % (result_dir, index),
               logits[index], fmt='%s')

    solver = VeriX(keras_model_path, x[index], y[index])
    solver.add_traversal_order('sensitivity_reversal')
    sat_set, unsat_set, timeout_set = solver.generate_explanation('sensitivity_reversal', epsilon=epsilon, timeout=timeout, verbosity=args.verbosity)

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
