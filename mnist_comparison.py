import os
import random
import argparse
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
from utils import plot_figure, plot_mask
from verix import VeriX

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=10)
parser.add_argument('--seed', type=int, default=137)
parser.add_argument('--traversal_mode', type=str, default='sensitivity_reversal')
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--path', type=str, default='mnist-models-comparison/')
args = parser.parse_args()

traversal_mode = args.traversal_mode
epsilon = args.epsilon
result_path = 'outputs/' + args.path

if not os.path.exists(result_path):
    os.mkdir(result_path)

# Load test data
random.seed(args.seed)
(_, _), (x_test, y_test) = mnist.load_data()
num_test = x_test.shape[0]
x_test = x_test.reshape(num_test, 28, 28, 1)
y_test = tf.keras.utils.to_categorical(y_test, 10)
x_test = x_test.astype('float32') / 255
# Choose samples to generate explanations for and plot
sample_indices = random.sample(range(0, num_test), args.num_samples)
x_sample = x_test[sample_indices]
y_sample = y_test[sample_indices]

# Load pre-trained networks
networks_path = 'networks/'
keras_models = []
model_names = []
for filename in os.listdir(networks_path):
    if filename.startswith('mnist') and filename.endswith('.h5'):
        onnx_path = networks_path + filename[:-3] + '.onnx'
        if not os.path.exists(onnx_path):
            continue
        keras_model_path = networks_path + filename
        keras_models.append(keras_model_path)
        model_names.append(filename[:-3])

# For each network, produce explanations on the set of images
for model_index in range(len(keras_models)):
    model_index = 2
    model_name = model_names[model_index]
    for sample_index in sample_indices:
        sample_index = 0
        original_image = x_test[sample_index]
        original_label = y_test[sample_index].argmax()
        result_sub_path = result_path + '%s-index-%d/' % (model_name, sample_index)

        # Generate explanation
        solver = VeriX(keras_models[model_index], original_image, y_test[sample_index], seed=args.seed)
        sensitivity = solver.add_traversal_order(traversal_mode)

        test = True  # in testing mode, read pre-stored explanation sets instead of actually generating explanations

        if not test:
            sat_set, unsat_set, timeout_set = solver.generate_explanation(traversal_mode, epsilon, 60, verbosity=False)
        else:
            sat_set = np.loadtxt(result_sub_path + 'index-%d-%s-linf%g-sat.txt' % (sample_index, model_name, epsilon),
                                 dtype=int)
            unsat_set = np.loadtxt(
                result_sub_path + 'index-%d-%s-linf%g-unsat.txt' % (sample_index, model_name, epsilon), dtype=int)
            timeout_set = np.loadtxt(
                result_sub_path + 'index-%d-%s-linf%g-timeout.txt' % (sample_index, model_name, epsilon), dtype=int)

        if not os.path.exists(result_sub_path):
            os.mkdir(result_sub_path)

        # Results for the sample
        if not test:
            with open(result_sub_path + 'summary.txt', 'w') as f:
                f.write('sat set size: %d\n' % len(sat_set))
                f.write('unsat set size: %d\n' % len(unsat_set))
                f.write('timeout set size: %d\n' % len(timeout_set))
            # explanation sets
            np.savetxt('%s/index-%d-%s-linf%g-unsat.txt' % (result_sub_path, sample_index, model_name, epsilon),
                       unsat_set, fmt='%d')
            np.savetxt('%s/index-%d-%s-linf%g-sat.txt' % (result_sub_path, sample_index, model_name, epsilon),
                       sat_set, fmt='%d')
            np.savetxt('%s/index-%d-%s-linf%g-timeout.txt' % (result_sub_path, sample_index, model_name, epsilon),
                       timeout_set, fmt='%d')
        # plot original input
        plot_figure(image=original_image,
                    path='%s/index-%d-original-%d-predicted-as-%d.png' %
                         (result_sub_path, sample_index, original_label, solver.pred),
                    cmap='gray')
        # plot sensitivity
        if sensitivity is not None:
            plot_figure(image=sensitivity.reshape(28, 28), path='%s/index-%d-%s-sensitivity.png' %
                                                                (result_sub_path, sample_index, model_name))
        # plot explanation
        plot_mask(original_image.reshape(28, 28), list(set(sat_set) | set(timeout_set)),
                  path='%s/index-%d-%s-linf%g-explanation.png' % (result_sub_path, sample_index, model_name, epsilon))
        plot_mask(original_image.reshape(28, 28), timeout_set,
                  path='%s/index-%d-%s-linf%g-timeout.png' % (result_sub_path, sample_index, model_name, epsilon))
