from maraboupy import Marabou
import pickle
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from skimage.color import label2rgb
from skimage.segmentation import mark_boundaries

def load_gtsrb(gtsrb_path):
    with open(gtsrb_path, 'rb') as handle:
        gtsrb = pickle.load(handle)

    # x_train, y_train = gtsrb['x_train'], gtsrb['y_train']
    # x_valid, y_valid = gtsrb['x_valid'], gtsrb['y_valid']
    x_test, y_test = gtsrb['x_test'], gtsrb['y_test']
    # x_train = x_train/255
    # x_valid = x_valid/255
    x_test = x_test/255
    # from keras.utils.np_utils import to_categorical
    # y_train = to_categorical(y_train)
    # y_valid = to_categorical(y_valid)
    # y_test = to_categorical(y_test)
    return x_test, y_test


def get_gtsrb_label(index):
    get_gtsrb_labels = ['30 mph', '50 mph', '60 mph', '70 mph',
                        '80 mph', '100 mph', '120 mph']
    return get_gtsrb_labels[index]


def plot_figure(image, path):
    fig = plt.figure()
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax = plt.Axes(fig, [-0.5, -0.5, 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(image)
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)

TIMEOUT = 60

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='gtsrb')
parser.add_argument('--network', type=str, default='gtsrb-30-20')
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--epsilon', type=float, default=0.1)
args = parser.parse_args()

dataset = args.dataset
model_name = args.network
index = args.index
epsilon = args.epsilon
# epsilon = 0.5

import os
result_dir = 'index-%d-%s-%ds-heuristic-linf%g' % (index, model_name, TIMEOUT, epsilon)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

gtsrb_path = 'GTSRB/gtsrb.pickle'
x_test, y_test = load_gtsrb(gtsrb_path=gtsrb_path)

from keras.models import load_model
# model_path = 'models/gtsrb-200x3'
model_path = 'models/' + model_name
keras_model = load_model(model_path + '.h5')
keras_model.summary()
# from keras.utils.np_utils import to_categorical
# score = keras_model.evaluate(x_test, to_categorical(y_test), verbose=1)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])

# index = 0
image = x_test[index]
label = y_test[index]
orig_label = get_gtsrb_label(index=label)
preds = keras_model.predict(np.expand_dims(image, axis=0))
prediction = np.argmax(preds)
pred_label = get_gtsrb_label(index=prediction)
# plt.imsave('gtsrb.png', image)
path = '%s/index-%d-original-%d-[%s]-predicted-as-%d-[%s].png' % (
    result_dir, index, label, orig_label, prediction, pred_label)
plot_figure(image, path)

if orig_label is not pred_label:
    exit()

onnx_model_path = 'models/' + model_name + '.onnx'
# mara_network = Marabou.read_onnx(onnx_model_path)
mara_network = Marabou.read_onnx(onnx_model_path,
                                 outputName=model_name + '/logit/BiasAdd:0')
options = Marabou.createOptions(numWorkers=16, timeoutInSeconds=TIMEOUT, verbosity=0)

# inputVars = mara_network.inputVars[0][0].flatten()
# outputVars = mara_network.outputVars.flatten()
inputVars = np.arange(32*32)
outputVars = mara_network.outputVars

# heuristic: get traverse order by pixel sensitivity
temp = image.reshape(32*32, 3)
image_batch = np.kron(np.ones((32*32, 1, 1)), temp)
image_batch_manipulated = image_batch.copy()
for i in range(32*32):
    # image_batch_manipulated[i][i] = 1 - image_batch_manipulated[i][i]
    image_batch_manipulated[i][i][:] = 0
# predictions = keras_model.predict(image_batch.reshape(32*32, 32, 32, 3))
# predictions_manipulated = keras_model.predict(image_batch_manipulated.reshape(32*32, 32, 32, 3))
from keras import backend
func = backend.function([keras_model.layers[0].input],
                        [keras_model.layers[keras_model.layers.__len__() - 1].input])
predictions = func([image_batch.reshape(32*32, 32, 32, 3)])[0]
predictions_manipulated = func([image_batch_manipulated.reshape(32*32, 32, 32, 3)])[0]
difference = predictions - predictions_manipulated
features = difference[:, label]
sorted_index = features.argsort()
inputVars = sorted_index
# inputVars = sorted_index[::-1]
#
sensitivity = features.reshape([32, 32])
path = '%s/index-%d-%s-sensitivity.png' % (
    result_dir, index, model_name)
plot_figure(sensitivity, path)

import random
random.shuffle(inputVars)
print(inputVars)


# image = x_test[index].flatten()
image = image.reshape(32*32, 3)
unsat_set = []
sat_set = []
timeout_set = []


for pixel in inputVars:
    for j in range(7):
        if j != label:
            # network = Marabou.read_onnx(onnx_model_path)
            network = Marabou.read_onnx(onnx_model_path,
                                        outputName=model_name + '/logit/BiasAdd:0')
            network.addInequality([outputVars[label], outputVars[j]],
                                  [1, -1], -1e-6)
            # network.addInequality([outputVars[label], outputVars[j]],
            #                       [1, -1], 0)
            for i in inputVars:
                if i == pixel or i in unsat_set:
                    network.setLowerBound(3 * i, max(0, image[i][0] - epsilon))
                    network.setUpperBound(3 * i, min(1, image[i][0] + epsilon))
                    network.setLowerBound(3 * i + 1, max(0, image[i][1] - epsilon))
                    network.setUpperBound(3 * i + 1, min(1, image[i][1] + epsilon))
                    network.setLowerBound(3 * i + 2, max(0, image[i][2] - epsilon))
                    network.setUpperBound(3 * i + 2, min(1, image[i][2] + epsilon))
                else:
                    network.setLowerBound(3 * i, image[i][0])
                    network.setUpperBound(3 * i, image[i][0])
                    network.setLowerBound(3 * i + 1, image[i][1])
                    network.setUpperBound(3 * i + 1, image[i][1])
                    network.setLowerBound(3 * i + 2, image[i][2])
                    network.setUpperBound(3 * i + 2, image[i][2])

            exitCode, vals, stats = network.solve(options=options)
            if exitCode == 'sat' or exitCode == 'TIMEOUT':
                break
            elif exitCode == 'unsat':
                continue

    if exitCode == 'unsat':
        print('location %d returns unsat, move out.' % pixel)
        unsat_set.append(pixel)
        print('current outside', unsat_set)
    elif exitCode == 'TIMEOUT':
        print('timeout for pixel', pixel)
        print('do not move out, continue to the next pixel')
        timeout_set.append(pixel)
    elif exitCode == 'sat':
        print('perturbing current outside + this location %d alters prediction' % pixel)
        print('do not move out, continue to the next pixel')
        sat_set.append(pixel)

        # adversary = [vals.get(i) for i in inputVars] ???????
        adversary = [vals.get(i) for i in mara_network.inputVars[0][0].flatten()]
        adversary = np.asarray(adversary).reshape(32, 32, 3)
        adv_predictions = [vals.get(i) for i in outputVars]
        adv_prediction = np.asarray(adv_predictions).argmax()
        adv_label = get_gtsrb_label(adv_prediction)
        path = '%s/index-%d-adversary-sat-pixel-%d-predicted-as-%d-[%s].png' % (
            result_dir, index, pixel, adv_prediction, adv_label)
        plot_figure(adversary, path)

    if pixel == inputVars[-1]:
        # mask = np.zeros(image.shape)
        mask = np.zeros(inputVars.shape).astype(bool)
        # mask[unsat_set] = 1
        mask[sat_set] = True
        mask[timeout_set] = True
        path = '%s/index-%d-%s-linf%g-explanation-%d-grey.png' % (
            result_dir, index, model_name, epsilon, len(sat_set)+len(timeout_set))
        plot_figure(label2rgb(mask.reshape(32, 32), x_test[index], bg_label=0), path)
        path = '%s/index-%d-%s-linf%g-explanation-%d-colour.png' % (
            result_dir, index, model_name, epsilon, len(sat_set)+len(timeout_set))
        plot_figure(label2rgb(mask.reshape(32, 32), x_test[index],
                              # colors=[[1, 1, 0]],
                              colors=[[0, 1, 0]],
                              bg_label=0,
                              saturation=1),
                    path)
        # plot_figure(mark_boundaries(x_test[index], mask.reshape(32, 32), mode='inner'), path)

        # mask = np.zeros(image.shape)
        mask = np.zeros(inputVars.shape).astype(bool)
        mask[timeout_set] = True
        path = '%s/index-%d-%s-linf%g-timeout-%d-grey.png' % (
            result_dir, index, model_name, epsilon, len(timeout_set))
        plot_figure(label2rgb(mask.reshape(32, 32), x_test[index], bg_label=0), path)
        path = '%s/index-%d-%s-linf%g-timeout-%d-colour.png' % (
            result_dir, index, model_name, epsilon, len(timeout_set))
        plot_figure(label2rgb(mask.reshape(32, 32), x_test[index],
                              # colors=[[1, 1, 0]],
                              colors=[[0, 1, 0]],
                              bg_label=0,
                              saturation=1),
                    path)
        # plot_figure(mark_boundaries(x_test[index], mask.reshape(32, 32), mode='inner'), path)

        np.savetxt('%s/index-%d-%s-timeout%d-linf%g-unsat.txt' % (
            result_dir, index, model_name, TIMEOUT, epsilon),
                   X=unsat_set, fmt='%d')
        np.savetxt('%s/index-%d-%s-timeout%d-linf%g-sat.txt' % (
            result_dir, index, model_name, TIMEOUT, epsilon),
                   X=sat_set, fmt='%d')
        np.savetxt('%s/index-%d-%s-timeout%d-linf%g-timeout.txt' % (
            result_dir, index, model_name, TIMEOUT, epsilon),
                   X=timeout_set, fmt='%d')

        # random_explanation = result_dir+'/'+result_dir+'.txt'
        # with open(random_explanation, 'a') as f:
        #     f.write(str(len(sat_set))+'\n')

print('explanation generated.')








