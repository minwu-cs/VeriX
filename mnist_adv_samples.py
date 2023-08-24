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
parser.add_argument('--adv_samples', type=str, default='mnist-30x2-normal-pgd-samples')
parser.add_argument('--network', type=str, default='mnist-10x2-normal')
parser.add_argument('--epsilon', type=float, default=0.05)
parser.add_argument('--timeout', type=int, default=60)
parser.add_argument('--verbosity', type=bool, default=False)
parser.add_argument('--th', type=int, default=5)
args = parser.parse_args()

dataset = args.dataset
model_name = args.network
epsilon = args.epsilon
timeout = args.timeout
adv_samples_path = 'adv_samples/' + args.adv_samples + '.npy'
output_path = 'outputs/' + args.adv_samples + '/'
if not os.path.exists(output_path):
    os.mkdir(output_path)
output_path = output_path + model_name + '/'
if not os.path.exists(output_path):
    os.mkdir(output_path)


(x_train, y_train), (x_test_real, y_test) = mnist.load_data()
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test_real = x_test_real.reshape(x_test_real.shape[0], 28, 28, 1)
# y_train = tf.keras.utils.to_categorical(y_train, 10)
# Do not divide by 255 because adv samples are generated from (0, 1) clipped values already
x_test = np.load(adv_samples_path)
y_test = tf.keras.utils.to_categorical(y_test, 10)[:len(x_test)]
# x_train = x_train.astype('float32') / 255
x_test_real = x_test_real.astype('float32') / 255

keras_model_path = directory + model_name + '.h5'
keras_model = load_model(keras_model_path)
logits = keras_model.predict(x_test)
preds = np.argmax(logits, axis=1)
correct_preds = np.argmax(y_test, axis=1)
incorrect_indices = np.where(preds != correct_preds)[0]
correct_indices = np.where(preds == correct_preds)[0]

# for pixel attack, take only samples with <th pixels perturbed
if 'pixel-attack' in args.adv_samples:
    keep_indices = np.where(np.sum(x_test - x_test_real[:len(x_test)] != 0, axis=(1, 2)) < 6)[0]
    incorrect_indices = np.intersect1d(keep_indices, incorrect_indices)
    correct_indices = np.intersect1d(keep_indices, correct_indices)

np.savetxt('%s%s-incorrect-indices.txt' % (output_path, model_name), incorrect_indices, fmt='%d')
np.savetxt('%s%s-correct-indices.txt' % (output_path, model_name), correct_indices, fmt='%d')

indices = np.concatenate((incorrect_indices[:100], correct_indices[:100]))
for index in indices:
    result_dir = '%sindex-%d-%s-%ds-heuristic-linf%g' % (output_path, index, model_name, timeout, epsilon)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    plot_figure(image=x_test[index],
                path='%s/index-%d-original-%d-predicted-as-%d.png' %
                     (result_dir, index, correct_preds[index], preds[index]),
                cmap='gray')
    np.savetxt('%s/index-%d-logits.txt' % (result_dir, index),
               logits[index], fmt='%s')

    solver = VeriX(keras_model_path, x_test[index], y_test[index])
    solver.add_traversal_order('sensitivity_reversal')
    sat_set, unsat_set, timeout_set = solver.generate_explanation('sensitivity_reversal', epsilon=epsilon, timeout=timeout, verbosity=args.verbosity)

    image = x_test[solver.pred].flatten()
    mask = np.zeros(image.shape).astype(bool)
    mask[sat_set] = True
    mask[timeout_set] = True

    plot_figure(image=label2rgb(mask.reshape(28, 28), x_test[index].reshape(28, 28),
                                # colors=[[1, 1, 0]],
                                # colors=[[128 / 255, 1, 0]],
                                colors=[[0, 1, 0]],
                                bg_label=0),
                path='%s/index-%d-%s-linf%g-explanation-%d.png' %
                     (result_dir, index, model_name, epsilon, len(sat_set)+len(timeout_set)))

    mask = np.zeros(image.shape).astype(bool)
    mask[timeout_set] = True

    plot_figure(image=label2rgb(mask.reshape(28, 28), x_test[index].reshape(28, 28),
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
