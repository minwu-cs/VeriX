import os
import argparse
import gzip
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import keras.datasets.cifar10 as cifar10
import cv2
from utils import  plot_figure
from verix import VeriX

np.random.seed(137)

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, default='mnist-10x2-normal')
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--timeout', type=int, default=60)
parser.add_argument('--verbosity', type=bool, default=False)
parser.add_argument('--dataset', type=str, default='emnist-letters')
args = parser.parse_args()

model_name = args.network
epsilon = args.epsilon
timeout = args.timeout
dataset = args.dataset

networks_directory = 'networks/'
keras_model_path = networks_directory + model_name + '.h5'

output_path = 'outputs/ood-samples/' + dataset
if not os.path.exists(output_path):
    os.mkdir(output_path)

# load data
num_samples = 100
if dataset == 'emnist-letters':
    f = gzip.open('emnist/emnist-letters-test-images-idx3-ubyte.gz', 'r')

    image_size = 28
    num_images = 20800

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    x_test = np.frombuffer(buf, dtype=np.uint8).astype(np.float32) / 255
    x_test = x_test.reshape(num_images, image_size, image_size)
    x_test = np.transpose(x_test, (0, 2, 1))

    f = gzip.open('emnist/emnist-letters-test-labels-idx1-ubyte.gz', 'r')
    f.read(8)
    buf = f.read(num_images)
    y_test = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    indices = np.random.randint(0, 20801, size=num_samples)
    x_test = x_test[indices]
    y_test = y_test[indices]

elif dataset == 'cifar-10':
    (_, _), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype(np.float32) / 255

    # crop images from 32x32 to 28x28
    x_test = x_test[:num_samples, 2:-2, 2:-2, :]
    y_test = y_test[:num_samples]

    #convert to grayscale
    gray_images = []
    for image in x_test:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_images.append(gray_image)
    x_test = np.array(gray_images)

    indices = np.arange(100)

else:
    print('invalid input for samples')
    exit()

for i in range(num_samples):
    index = indices[i]

    result_dir = ('%s/%s-index-%d-%s-%ds-heuristic-linf%g' %
                  (output_path, dataset, index, model_name, timeout, epsilon))
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # plot original
    plot_figure(image=x_test[i],
                path='%s/%s-index-%d-original-%d.png' % (result_dir, dataset, index, y_test[i]),
                cmap='gray')

    # generate explanation
    solver = VeriX(keras_model_path, x_test[i], y_test[i])
    sensitivities = solver.get_pixel_sensitivities(transformation=lambda a: 1 - a,
                                                   plot_path='%s/%s-index-%d-%s-sensitivity.png' %
                                                             (result_dir, dataset, index, model_name))
    solver.add_traversal_order('sensitivity_reversal')
    sat_set, unsat_set, timeout_set = solver.generate_explanation('sensitivity_reversal', epsilon=epsilon,
                                                                  timeout=timeout, verbosity=args.verbosity)

    # plot sensitivities & explanation
    image = x_test[solver.pred].flatten()
    mask = np.zeros(image.shape).astype(bool)
    mask[sat_set] = True
    mask[timeout_set] = True
    plot_figure(image=label2rgb(mask.reshape(28, 28), x_test[i].reshape(28, 28),
                                colors=[[0, 1, 0]],
                                bg_label=0),
                path='%s/%s-index-%d-%s-linf%g-explanation-%d.png' %
                     (result_dir, dataset, index, model_name, epsilon, len(sat_set) + len(timeout_set)))
    mask = np.zeros(image.shape).astype(bool)
    mask[timeout_set] = True
    plot_figure(image=label2rgb(mask.reshape(28, 28), x_test[i].reshape(28, 28),
                                colors=[[0, 1, 0]],
                                bg_label=0),
                path='%s/%s-index-%d-%s-linf%g-timeout-%d.png' %
                     (result_dir, dataset, index, model_name, epsilon, len(timeout_set)))

    # save explanation
    np.savetxt('%s/%s-index-%d-%s-linf%g-unsat.txt' % (result_dir, dataset, index, model_name, epsilon),
               unsat_set, fmt='%d')
    np.savetxt('%s/%s-index-%d-%s-linf%g-sat.txt' % (result_dir, dataset, index, model_name, epsilon),
               sat_set, fmt='%d')
    np.savetxt('%s/%s-index-%d-%s-linf%g-timeout.txt' % (result_dir, dataset, index, model_name, epsilon),
               timeout_set, fmt='%d')
