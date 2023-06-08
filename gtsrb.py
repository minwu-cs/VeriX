import sys
import time
import os
from maraboupy import Marabou
# from maraboupy import MarabouCore
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
    get_gtsrb_labels = ['50 mph', '30 mph', 'yield', 'priority road',
                        'keep right', 'no passing for large vechicles', '70 mph', '80 mph',
                        'road work', 'no passing']
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

TIMEOUT = 300

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='gtsrb')
parser.add_argument('--network', type=str, default='gtsrb-10x2')
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--epsilon', type=float, default=0.01)
parser.add_argument('--traverse', type=str, default='heuristic')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

dataset = args.dataset
model_name = args.network
index = args.index
epsilon = args.epsilon
traverse = args.traverse
seed = args.seed


if traverse == 'heuristic':
    result_dir = 'index-%d-%s-%ds-%s-linf%g' % (
        index, model_name, TIMEOUT, traverse, epsilon)
elif traverse == 'random':
    result_dir = 'index-%d-%s-%ds-%s-seed-%d-linf%g' % (
        index, model_name, TIMEOUT, traverse, seed, epsilon)
else:
    print('traversal incorrect.')
    exit()

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

# if label in [0,1,6,7]:
#     print("Not ideal label")
#     exit()

orig_label = get_gtsrb_label(index=label)
preds = keras_model.predict(np.expand_dims(image, axis=0))
prediction = np.argmax(preds)
pred_label = get_gtsrb_label(index=prediction)
# plt.imsave('gtsrb.png', image)
path = '%s/index-%d-original-%d-[%s]-predicted-as-%d-[%s].png' % (
    result_dir, index, label, orig_label, prediction, pred_label)
plot_figure(image, path)

if orig_label is not pred_label:
    print("Misclassify")
    exit()


# explanation_tick = time.time()

onnx_model_path = 'models/' + model_name + '.onnx'
# mara_network = Marabou.read_onnx(onnx_model_path)
mara_network = Marabou.read_onnx(onnx_model_path,
                                 outputName=model_name + '/logit/BiasAdd:0')
options = Marabou.createOptions(numWorkers=16, timeoutInSeconds=TIMEOUT, verbosity=0, solveWithMILP=True)

# inputVars = mara_network.inputVars[0][0].flatten() # channel causes error
# outputVars = mara_network.outputVars.flatten()
inputVars = np.arange(32*32)
outputVars = mara_network.outputVars.flatten()

if traverse == 'heuristic':
    saliency_tic = time.time()
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

    np.savetxt('%s/index-%d-%s-linf%g-saliency.txt' % (result_dir, index, model_name, epsilon),
               np.flip(inputVars), fmt='%d')

    sensitivity = features.reshape([32, 32])
    path = '%s/index-%d-%s-sensitivity.png' % (
        result_dir, index, model_name)
    plot_figure(sensitivity, path)

    exit()

    saliency_toc = time.time()
    saliency_time = saliency_toc - saliency_tic

elif traverse == 'random':
    import random
    random.seed(seed)
    random.shuffle(inputVars)


# print(inputVars.shape)
# print(inputVars)

# exit()

# image = x_test[index].flatten()
image = image.reshape(32*32, 3)
unsat_set = []
sat_set = []
timeout_set = []

MIN_EPS = args.epsilon
robust = True
for j in range(10):
    if j != label:
        network = Marabou.read_onnx(onnx_model_path,
                                    outputName=model_name + '/logit/BiasAdd:0')
        network.addInequality([outputVars[label], outputVars[j]],
                              [1, -1], -1e-6)

        for i in inputVars:
            network.setLowerBound(3 * i, max(0, image[i][0] - MIN_EPS))
            network.setUpperBound(3 * i, min(1, image[i][0] + MIN_EPS))
            network.setLowerBound(3 * i + 1, max(0, image[i][1] - MIN_EPS))
            network.setUpperBound(3 * i + 1, min(1, image[i][1] + MIN_EPS))
            network.setLowerBound(3 * i + 2, max(0, image[i][2] - MIN_EPS))
            network.setUpperBound(3 * i + 2, min(1, image[i][2] + MIN_EPS))

        exitCode, vals, stats = network.solve(options=options, verbose=False)

        if exitCode == 'unsat':
            continue
        elif exitCode == 'sat':
            robust = False
            break
        else:
            print("timeout when checking adv robustness!")
            exit(0)
if robust:
    print("Adv. robust. ")
    exit(0)

# network = Marabou.read_onnx(onnx_model_path,
#                             outputNames=[model_name + '/logit/BiasAdd:0'])


marabou_time = []
explanation_tick = time.time()

for pixel in inputVars:
    # print("current pixel: ", pixel)
    p_path = '%s/index-%d-pixel-%d.txt' % (result_dir, index, pixel)

    cached = False
    if os.path.isfile(p_path):
        exitCode = open(p_path).read().strip()
        cached = True
    else:
        for j in range(10):
            if j != label:
                network = Marabou.read_onnx(onnx_model_path,
                                            outputName=model_name + '/logit/BiasAdd:0')

                # network.addInequality([outputVars[label], outputVars[j]],
                #                       [1, -1], -1e-6, isProperty=True)
                network.addInequality([outputVars[label], outputVars[j]],
                                      [1, -1], -1e-6)

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
                marabou_tick = time.time()
                exitCode, vals, stats = network.solve(options=options, verbose=False)
                marabou_toc = time.time()
                marabou_time.append(marabou_toc - marabou_tick)

                # network.clearProperty()

                if exitCode == 'sat' or exitCode == 'TIMEOUT':
                    break
                elif exitCode == 'unsat':
                    continue
        with open(p_path, 'w') as out_file:
            out_file.write(exitCode)
    
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

        # if not cached:
        #     # adversary = [vals.get(i) for i in inputVars] ???????
        #     adversary = [vals.get(i) for i in mara_network.inputVars[0][0].flatten()]
        #     adversary = np.asarray(adversary).reshape(32, 32, 3)
        #     adv_predictions = [vals.get(i) for i in outputVars]
        #     adv_prediction = np.asarray(adv_predictions).argmax()
        #     adv_label = get_gtsrb_label(adv_prediction)
        #     path = '%s/index-%d-adversary-sat-pixel-%d-predicted-as-%d-[%s].png' % (
        #         result_dir, index, pixel, adv_prediction, adv_label)
        #     # plot_figure(adversary, path)

    explanation_toc = time.time()
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
        colors = [[0, 1, 0]] if traverse == 'heuristic' else [[1, 0, 0]]
        plot_figure(label2rgb(mask.reshape(32, 32), x_test[index],
                              # colors=[[1, 1, 0]],
                              # colors=[[0, 1, 0]],
                              colors=colors,
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

marabou_time = np.asarray(marabou_time)
marabou_time = np.mean(marabou_time)
if traverse == "heuristic":
    explanation_time = explanation_toc - explanation_tick + saliency_time
elif traverse == "random":
    explanation_time = explanation_toc - explanation_tick

marabou_time_text = '%s-%ds-%s-linf%g-marabou-time.txt' % (model_name, TIMEOUT, traverse, epsilon)
with open(marabou_time_text, 'a') as f:
    f.write(str(marabou_time) + '\n')

explanation_time_text = '%s-%ds-%s-linf%g-explanation-time.txt' % (model_name, TIMEOUT, traverse, epsilon)
with open(explanation_time_text, 'a') as f:
    f.write(str(explanation_time) + '\n')

explanation_size_text = '%s-%ds-%s-linf%g-explanation-size.txt' % (model_name, TIMEOUT, traverse, epsilon)
with open(explanation_size_text, 'a') as f:
    f.write(str(len(sat_set)+len(timeout_set)) + '\n')

print('explanation generated.')








