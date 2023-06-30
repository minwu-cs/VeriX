import os, sys
import random
import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr
from keras.models import load_model
from maraboupy import Marabou
from utils import suppress_stdout, plot_figure
import warnings

# TODO: add support for adversary

class VeriX:
    def __init__(self, network_path, x, y, loss=tfr.keras.losses.SoftmaxLoss(), optimizer=tf.keras.optimizers.Adam(),
                 metrics=None, seed=137):
        """
        Initializes solver object and prints a summary of keras model loaded
        :param network_path: path to the .h5 network
        :param x: input image
        :param y: correct prediction class label
        """
        # Should actually convert from h5 to onnx here, but I have trouble using tf2onnx on M1 mac,
        # so I will instead try to find an onnx model with the same name under the same path as the h5 model
        self.onnx_path = network_path[:-3] + '.onnx'
        if not os.path.exists(self.onnx_path):
            raise Exception("onnx model not found")
        if metrics is None:
            metrics = ['accuracy']
        self.keras_model = load_model(network_path)
        self.keras_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        self.keras_model.summary()
        self.mara_network = Marabou.read_onnx(self.onnx_path)
        self.x = x
        self.y = y
        self.num_pixels = x.shape[0] * x.shape[1]
        self.x_flat = x.reshape((self.num_pixels, -1))
        self.logits = self.keras_model.predict(np.expand_dims(self.x, axis=0))
        self.pred = self.logits.argmax()
        print('logits:', self.logits[0])
        print('prediction:', self.pred)
        self.traversal_orders = {}
        random.seed(seed)

    def add_traversal_order(self, mode='sensitivity_deletion', order=None):
        """
        If order is not none, add the user-defined traversal order
        Otherwise, compute the pixel traversal order with one of the default methods
        :param mode: mode of traversal
        :param order: optional user-defined traversal order
        :return: if using default sensitivity traversal, returns pixel sensitivities
        """
        if mode in self.traversal_orders.keys():
            warnings.warn(f'Traversal order by {mode} already exists, overwriting.')

        if order is not None:
            # add user-defined order
            if order.shape != (self.num_pixels,):
                raise Exception(
                    f'Invalid traversal order given: Expected array of shape ({self.num_pixels},), got array '
                    f'of shape {order.shape}.')
            self.traversal_orders[mode] = order

        else:
            # compute with default methods
            if mode == 'sensitivity_deletion':
                sensitivity = self.get_pixel_sensitivities(transformation=lambda a: 0)
                self.traversal_orders[mode] = np.argsort(sensitivity)
                return sensitivity

            if mode == 'sensitivity_reversal':
                sensitivity = self.get_pixel_sensitivities(transformation=lambda a: 1 - a)
                self.traversal_orders[mode] = np.argsort(sensitivity)
                return sensitivity

            elif mode == 'random':
                self.traversal_orders[mode] = np.arange(self.num_pixels)
                random.shuffle(self.traversal_orders[mode])

            else:
                raise Exception('Invalid mode')

    def generate_explanation(self, traversal_mode, epsilon, timeout=60, verbosity=True):
        unsat_set = []
        sat_set = []
        timeout_set = []
        output_vars = self.mara_network.outputVars[0].flatten()
        if traversal_mode not in self.traversal_orders.keys():
            raise Exception('Invalid traversal order!')
        traversal_order = self.traversal_orders[traversal_mode]
        options = Marabou.createOptions(numWorkers=16, timeoutInSeconds=timeout, verbosity=0)

        progress = 0
        end = '\n' if verbosity else ''
        print('Progress: %d/%d pixels checked' % (progress, len(traversal_order)), end=end)
        num_channels = self.x_flat.shape[1]
        for pixel in traversal_order:
            for j in range(len(output_vars)):
                if j != self.pred:
                    network = Marabou.read_onnx(self.onnx_path)
                    network.addInequality([output_vars[self.pred], output_vars[j]],
                                          [1, -1], -1e-6)
                    for i in traversal_order:
                        for channel in range(num_channels):
                            variable_number = num_channels * i + channel
                            if i == pixel or i in unsat_set:
                                network.setLowerBound(variable_number, max(0, self.x_flat[i][channel] - epsilon))
                                network.setUpperBound(variable_number, min(1, self.x_flat[i][channel] + epsilon))
                            else:
                                network.setLowerBound(variable_number, self.x_flat[i][channel])
                                network.setUpperBound(variable_number, self.x_flat[i][channel])

                    with suppress_stdout():  # ugly but working fix to stop gurobi liscence info from printing
                        exit_code, vals, stats = network.solve(options=options, verbose=False)

                    if exit_code == 'sat' or exit_code == 'TIMEOUT':
                        break
                    elif exit_code == 'unsat':
                        continue

            # if verbosity is false, redirect stdout to none
            original_stdout = sys.stdout
            if not verbosity:
                sys.stdout = None

            # add pixel to correct set
            if exit_code == 'unsat':
                print('location %d returns unsat, move out.' % pixel)
                unsat_set.append(pixel)
            elif exit_code == 'TIMEOUT':
                print('timeout for pixel', pixel)
                print('do not move out, continue to the next pixel')
                timeout_set.append(pixel)
            elif exit_code == 'sat':
                print('perturbing current outside + this location %d alters prediction' % pixel)
                print('do not move out, continue to the next pixel')
                sat_set.append(pixel)

            # if verbosity is false, restore stdout to original state
            if not verbosity:
                sys.stdout = original_stdout

            # display progress
            progress += 1
            print('\rProgress: %d/%d pixels checked' % (progress, len(traversal_order)), end=end)

        print('explanation generated.')

        return sat_set, unsat_set, timeout_set

    def get_pixel_sensitivities(self, transformation=lambda x: 0, plot_path=None):
        """
        Computes pixel sensitivities as a 1D array with number of pixels elements
        If plot path is specified, plots sensitivity heatmap
        :param transformation: specifies the x' = T(x) transformation
        :param plot_path: path to store sensitivity map png
        :return: pixel sensitivities
        """
        image_batch = np.repeat(self.x_flat[None, ...], self.num_pixels, axis=0)
        image_batch_manipulated = image_batch.copy()
        for i in range(self.num_pixels):
            image_batch_manipulated[i][i] = transformation(image_batch_manipulated[i][i])
        batch_shape = tuple([self.num_pixels]) + self.x.shape
        predictions = self.keras_model.predict(image_batch.reshape(batch_shape))
        predictions_manipulated = self.keras_model.predict(image_batch_manipulated.reshape(batch_shape))
        difference = predictions - predictions_manipulated
        sensitivity = difference[:, self.pred]

        if plot_path is not None:
            sensitivity_map = sensitivity.reshape(self.x.shape[:2])
            plot_figure(image=sensitivity_map, path=plot_path)
        return sensitivity
