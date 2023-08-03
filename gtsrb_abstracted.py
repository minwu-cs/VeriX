import sys
import time
import os
import pickle
import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from skimage.color import label2rgb
from verix import VeriX

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='gtsrb')
parser.add_argument('--network', type=str, default='gtsrb-10x2')
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--traverse', type=str, default='heuristic')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--timeout', type=int, default=300)
parser.add_argument('--verbosity', type=bool, default=False)
parser.add_argument('--output_path', type=str, default='gtsrb_outputs')
args = parser.parse_args()

dataset = args.dataset
model_name = args.network
index = args.index
epsilon = args.epsilon
traverse = args.traverse
seed = args.seed
timeout = args.timeout
output_path = args.output_path
if not os.path.exists(output_path):
    os.mkdir(output_path)


def load_gtsrb(gtsrb_path):
    with open(gtsrb_path, 'rb') as handle:
        gtsrb = pickle.load(handle)

    # x_train, y_train = gtsrb['x_train'], gtsrb['y_train']
    # x_valid, y_valid = gtsrb['x_valid'], gtsrb['y_valid']
    x_test, y_test = gtsrb['x_test'], gtsrb['y_test']
    # x_train = x_train/255
    # x_valid = x_valid/255
    x_test = x_test / 255
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

if traverse == 'heuristic':
    result_dir = '%s/index-%d-%s-%ds-%s-linf%g' % (output_path, index, model_name, timeout, traverse, epsilon)
elif traverse == 'random':
    result_dir = '%s/index-%d-%s-%ds-%s-seed-%d-linf%g' % (output_path, index, model_name, timeout, traverse, seed, epsilon)
else:
    print('traversal incorrect.')
    exit()

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

gtsrb_path = 'train_networks/gtsrb.pickle'
x_test, y_test = load_gtsrb(gtsrb_path=gtsrb_path)

from keras.models import load_model

model_path = 'networks/' + model_name
keras_model = load_model(model_path + '.h5')
keras_model.summary()
image = x_test[index]
label = y_test[index]

orig_label = get_gtsrb_label(index=label)
preds = keras_model.predict(np.expand_dims(image, axis=0))
prediction = np.argmax(preds)
pred_label = get_gtsrb_label(index=prediction)
path = '%s/index-%d-original-%d-[%s]-predicted-as-%d-[%s].png' % (
    result_dir, index, label, orig_label, prediction, pred_label)
plot_figure(image, path)

keras_path = 'networks/' + model_name + '.h5'
solver = VeriX(keras_path, x_test[index], y_test[index], seed=seed)
solver.add_traversal_order('sensitivity_deletion')
sat_set, unsat_set, timeout_set = solver.generate_explanation('sensitivity_deletion', epsilon, verbosity=args.verbosity)

np.savetxt('%s/index-%d-%s-timeout%d-linf%g-unsat.txt' % (
    result_dir, index, model_name, timeout, epsilon),
           X=unsat_set, fmt='%d')
np.savetxt('%s/index-%d-%s-timeout%d-linf%g-sat.txt' % (
    result_dir, index, model_name, timeout, epsilon),
           X=sat_set, fmt='%d')
np.savetxt('%s/index-%d-%s-timeout%d-linf%g-timeout.txt' % (
    result_dir, index, model_name, timeout, epsilon),
           X=timeout_set, fmt='%d')

# image = x_test[index].flatten()
mask = np.zeros(solver.x_flat.shape[0]).astype(bool)
image = image.reshape(32 * 32, 3)

# mask[unsat_set] = 1
mask[sat_set] = True
mask[timeout_set] = True
path = '%s/index-%d-%s-linf%g-explanation-%d-grey.png' % (
    result_dir, index, model_name, epsilon, len(sat_set) + len(timeout_set))
plot_figure(label2rgb(mask.reshape(32, 32), x_test[index], bg_label=0, colors=[[0, 1, 0]]), path)
path = '%s/index-%d-%s-linf%g-explanation-%d-colour.png' % (
    result_dir, index, model_name, epsilon, len(sat_set) + len(timeout_set))
colors = [[0, 1, 0]] if traverse == 'heuristic' else [[1, 0, 0]]
plot_figure(label2rgb(mask.reshape(32, 32), x_test[index],
                      colors=colors,
                      bg_label=0,
                      saturation=1),
            path)
# plot_figure(mark_boundaries(x_test[index], mask.reshape(32, 32), mode='inner'), path)

# mask = np.zeros(image.shape)
mask = np.zeros(mask.shape).astype(bool)
mask[timeout_set] = True
path = '%s/index-%d-%s-linf%g-timeout-%d-grey.png' % (
    result_dir, index, model_name, epsilon, len(timeout_set))
plot_figure(label2rgb(mask.reshape(32, 32), x_test[index], bg_label=0, colors=[[0, 1, 0]]), path)
path = '%s/index-%d-%s-linf%g-timeout-%d-colour.png' % (
    result_dir, index, model_name, epsilon, len(timeout_set))
plot_figure(label2rgb(mask.reshape(32, 32), x_test[index],
                      # colors=[[1, 1, 0]],
                      colors=[[0, 1, 0]],
                      bg_label=0,
                      saturation=1),
            path)
# plot_figure(mark_boundaries(x_test[index], mask.reshape(32, 32), mode='inner'), path)

# random_explanation = result_dir+'/'+result_dir+'.txt'
# with open(random_explanation, 'a') as f:
#     f.write(str(len(sat_set))+'\n')

print('explanation generated.')
