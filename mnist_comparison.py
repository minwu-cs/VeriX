import os
import random
import argparse
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
from utils import plot_figure, plot_mask
from verix import VeriX
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=5)
parser.add_argument('--seed', type=int, default=137)
parser.add_argument('--traversal_mode', type=str, default='sensitivity_reversal')
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--path', type=str, default='mnist-models-comparison/')
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--name', type=str, default='explanation_comparisons')
args = parser.parse_args()

traversal_mode = args.traversal_mode
epsilon = args.epsilon
result_path = 'outputs/' + args.path
num_samples = args.num_samples

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
sample_indices = random.sample(range(0, num_test), num_samples)
x_sample = x_test[sample_indices]
y_sample = y_test[sample_indices]

# Load pre-trained networks
# networks_path = 'networks/'
# keras_models = []
# model_names = []
# for filename in os.listdir(networks_path):
#     if filename.startswith('mnist') and filename.endswith('.h5'):
#         onnx_path = networks_path + filename[:-3] + '.onnx'
#         if not os.path.exists(onnx_path):
#             continue
#         keras_model_path = networks_path + filename
#         keras_models.append(keras_model_path)
#         model_names.append(filename[:-3])

networks_path = 'networks/'
keras_models = []
model_names = ['mnist-10x2', 'mnist-simple-cnn']
for model_name in model_names:
    keras_model_path = networks_path + model_name + '.h5'
    onnx_model_path = networks_path + model_name + '.onnx'
    if not os.path.exists(onnx_model_path) and os.path.exists(keras_model_path):
        raise Exception('model not found')
    keras_models.append(keras_model_path)

# For each network, produce explanations on the set of images
results = {}
num_models = len(keras_models)
fig, axes = plt.subplots(num_samples, num_models)
for model_index in range(num_models):
    model_name = model_names[model_index]
    results[model_name] = {}
    for i in range(len(sample_indices)):
        sample_index = sample_indices[i]
        original_image = x_test[sample_index]
        original_label = y_test[sample_index].argmax()
        result_sub_path = result_path + '%s-index-%d/' % (model_name, sample_index)

        test = args.test
        # in testing mode, read pre-stored explanation sets instead of actually generating explanations

        if not test:
            # Generate explanation
            solver = VeriX(keras_models[model_index], original_image, y_test[sample_index], seed=args.seed)
            sensitivity = solver.add_traversal_order(traversal_mode)
            print("Generating explanation for sample " + str(sample_index) + ", network " + model_name)
            sat_set, unsat_set, timeout_set = solver.generate_explanation(traversal_mode, epsilon, 60, verbosity=False)
        else:
            solver = VeriX(keras_models[2], original_image, y_test[sample_index], seed=args.seed)
            sensitivity = solver.add_traversal_order(traversal_mode)
            load_path = result_path + '%s-index-%d/' % (model_names[2], 0)
            sat_set = np.loadtxt(load_path + 'index-%d-%s-linf%g-sat.txt' % (0, model_names[2], epsilon),
                                 dtype=int)
            unsat_set = np.loadtxt(
                load_path + 'index-%d-%s-linf%g-unsat.txt' % (0, model_names[2], epsilon), dtype=int)
            timeout_set = np.loadtxt(
                load_path + 'index-%d-%s-linf%g-timeout.txt' % (0, model_names[2], epsilon), dtype=int)
        results[model_name][sample_index] = (sat_set, unsat_set, timeout_set)

        if not os.path.exists(result_sub_path):
            os.mkdir(result_sub_path)

        # Results for the sample
        if not test:
            with open(result_sub_path + 'summary.txt', 'w') as f:
                f.write('model: %s\n' % model_name)
                f.write('index: %d\n' % sample_index)
                f.write('traversal mode: %s\n' % traversal_mode)
                f.write('epsilon: %f\n' % epsilon)
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
        masked_image = plot_mask(original_image.reshape(28, 28), list(set(sat_set) | set(timeout_set)),
                                 path='%s/index-%d-%s-linf%g-explanation.png' % (
                                     result_sub_path, sample_index, model_name, epsilon))
        plot_mask(original_image.reshape(28, 28), timeout_set,
                  path='%s/index-%d-%s-linf%g-timeout.png' % (result_sub_path, sample_index, model_name, epsilon))

        # plot on grid
        ax = axes[i, model_index]
        ax.axis('off')
        ax.imshow(masked_image, interpolation='nearest')

# save results in the same place
with open('results.pickle', 'wb') as f:
    pickle.dump(results, f)

# plot summary image
# row names
for i in range(num_samples):
    fig.text(0.03, 0.03 + (i + 0.5) / num_samples * 0.95, f'{sample_indices[i]}',
             ha='center', va='center', rotation='vertical')
# col names
for i in range(num_models):
    fig.text(0.03 + (i + 0.5) / num_models * 0.95, 0.97, model_names[i][6:], ha='center', va='center')

fig.tight_layout()
fig.subplots_adjust(top=0.95)
fig.savefig('%s/%s_linf%g_%s.png' % (result_path, traversal_mode, epsilon, args.name))
plt.close(fig)
