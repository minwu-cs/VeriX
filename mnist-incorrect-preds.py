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
from utils import suppress_stdout

def plot_figure(image, path, cmap=None):
    fig = plt.figure()
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax = plt.Axes(fig, [-0.5, -0.5, 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if cmap is None:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap=cmap)
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)


TIMEOUT = 60
directory = 'models/'
if not os.path.exists(directory):
    os.mkdir(directory)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--network', type=str, default='mnist-10x2')
# parser.add_argument('--index', type=int, default=0)
parser.add_argument('--epsilon', type=float, default=0.1)
args = parser.parse_args()

dataset = args.dataset
model_name = args.network
# index = args.index
epsilon = args.epsilon

# load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x = x_test
y = y_test

# load model
keras_model_path = directory + model_name + '.h5'
keras_model = load_model(keras_model_path)
keras_model.summary()
# keras_model.compile(loss=tfr.keras.losses.SoftmaxLoss(),
#                     optimizer=tf.keras.optimizers.Adam(),
#                     metrics=['accuracy'])
score = keras_model.evaluate(x, y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

for index in range(len(x_test)):
    correct_pred = y[index].argmax()
    logits = keras_model.predict(np.expand_dims(x[index], axis=0))
    label = logits.argmax()
    print(logits)
    print('%d predicted as %d', correct_pred, label)

    if label == y[index].argmax():
        print("Correct prediction. Continue.")
        continue
    print("Wrong prediction. Generate explanation.")

    result_dir = 'incorrect-mnist-outputs/index-%d-%s-%ds-heuristic-linf%g' % (index, model_name, TIMEOUT, epsilon)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    plot_figure(image=x[index],
                path='%s/index-%d-original-%d-predicted-as-%d.png' % (result_dir, index, correct_pred, label),
                cmap='gray')

    explanation_tick = time.time()

    # heuristic: get traverse order by pixel sensitivity
    temp = x[index].reshape(28*28)
    image_batch = np.kron(np.ones((28*28, 1)), temp)
    image_batch_manipulated = image_batch.copy()
    for i in range(28*28):
        image_batch_manipulated[i][i] = 1 - image_batch_manipulated[i][i]
        # image_batch_manipulated[i][i] = 0
    predictions = keras_model.predict(image_batch.reshape(784, 28, 28, 1))
    predictions_manipulated = keras_model.predict(image_batch_manipulated.reshape(784, 28, 28, 1))
    difference = predictions - predictions_manipulated
    features = difference[:, label]
    sorted_index = features.argsort()
    # inputVars = sorted_index

    sensitivity = features.reshape([28, 28])
    plot_figure(image=sensitivity,
                path='%s/index-%d-%s-sensitivity.png' % (result_dir, index, model_name))

    onnx_model_path = directory + model_name + '.onnx'
    mara_network = Marabou.read_onnx(onnx_model_path)
    options = Marabou.createOptions(numWorkers=16, timeoutInSeconds=TIMEOUT, verbosity=0)

    inputVars = mara_network.inputVars[0][0].flatten()
    outputVars = mara_network.outputVars[0].flatten()
    inputVars = sorted_index
    # print(inputVars)

    image = x[index].flatten()

    unsat_set = []
    sat_set = []
    timeout_set = []

    marabou_time = []
    for pixel in inputVars:
        for j in range(10):
            if j != label:
                network = Marabou.read_onnx(onnx_model_path)
                network.addInequality([outputVars[label], outputVars[j]],
                                      [1, -1], -1e-6)
                # network.addInequality([outputVars[label], outputVars[j]],
                #                       [1, -1], 0)
                for i in inputVars:
                    if i == pixel or i in unsat_set:
                        # network.setLowerBound(i, 0)
                        # network.setUpperBound(i, 1)
                        network.setLowerBound(i, max(0, image[i] - epsilon))
                        network.setUpperBound(i, min(1, image[i] + epsilon))
                    else:
                        network.setLowerBound(i, image[i])
                        network.setUpperBound(i, image[i])
                marabou_tick = time.time()
                with suppress_stdout():
                    exitCode, vals, stats = network.solve(options=options, verbose=False)
                marabou_toc = time.time()
                marabou_time.append(marabou_toc - marabou_tick)
                if exitCode == 'sat' or exitCode == 'TIMEOUT':
                    break
                elif exitCode == 'unsat':
                    continue

        if exitCode == 'unsat':
            # print('location %d returns unsat, move out.' % pixel)
            unsat_set.append(pixel)
            # print('current outside', unsat_set)
        elif exitCode == 'TIMEOUT':
            # print('timeout for pixel', pixel)
            # print('do not move out, continue to the next pixel')
            timeout_set.append(pixel)
        elif exitCode == 'sat':
            # print('perturbing current outside + this location %d alters prediction' % pixel)
            # print('do not move out, continue to the next pixel')
            sat_set.append(pixel)

            # adversary = [vals.get(i) for i in inputVars] ???????
            adversary = [vals.get(i) for i in mara_network.inputVars[0][0].flatten()]
            adversary = np.asarray(adversary).reshape(28, 28)
            prediction = [vals.get(i) for i in outputVars]
            prediction = np.asarray(prediction).argmax()

            # plot_figure(image=adversary,
            #             path='%s/index-%d-adversary-sat-pixel-%d-predicted-as-%d.png' %
            #                  (result_dir, index, pixel, prediction),
            #             cmap='gray')

        if pixel == inputVars[-1]:
            explanation_toc = time.time()

            mask = np.zeros(image.shape).astype(bool)
            # mask[unsat_set] = 1
            mask[sat_set] = True
            mask[timeout_set] = True
            # mask = mask.astype('int')

            plot_figure(image=label2rgb(mask.reshape(28, 28), x[index].reshape(28, 28),
                                        # colors=[[1, 1, 0]],
                                        # colors=[[128 / 255, 1, 0]],
                                        colors=[[0, 1, 0]],
                                        bg_label=0),
                        path='%s/index-%d-%s-linf%g-explanation-%d.png' %
                             (result_dir, index, model_name, epsilon, len(sat_set)+len(timeout_set)))

            mask = np.zeros(image.shape).astype(bool)
            mask[timeout_set] = True
            # mask = mask.astype('int')

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

    marabou_time = np.asarray(marabou_time)
    marabou_time = np.mean(marabou_time)
    explanation_time = explanation_toc - explanation_tick

    marabou_time_text = directory + '%s-%ds-heuristic-linf%g-marabou-time.txt' % (model_name, TIMEOUT, epsilon)
    with open(marabou_time_text, 'a') as f:
        f.write(str(marabou_time) + '\n')

    explanation_time_text = directory + '%s-%ds-heuristic-linf%g-explanation-time.txt' % (model_name, TIMEOUT, epsilon)
    with open(explanation_time_text, 'a') as f:
        f.write(str(explanation_time) + '\n')

    print('explanation generated.')

