import os
import random
import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr
from keras.models import load_model
from maraboupy import Marabou
from utils import suppress_stdout, plot_figure
import warnings


class VeriX:
    def __init__(self, network, x, y=None, loss=tfr.keras.losses.SoftmaxLoss(), optimizer=tf.keras.optimizers.Adam(),
                 metrics=None, seed=137):
        """
        :param network: path to the .h5 network
        :param x: input image
        :param y: correct prediction class label
        """
        # Should actually convert from h5 to onnx here, but I have trouble using tf2onnx on M1 mac,
        # so I will instead try to find an onnx model with the same name under the same path as the h5 model
        onnx_path = network[:-3] + '.onnx'
        if not os.path.exists(onnx_path):
            raise Exception("onnx model not found")
        if metrics is None:
            metrics = ['accuracy']
        self.keras_model = load_model(network)
        self.keras_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        self.keras_model.summary()
        self.mara_network = Marabou.read_onnx(onnx_path)
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

    def add_traversal_order(self, mode='sensitivity', order=None):
        """
        If order is not none, add the user-defined traversal order
        Otherwise, compute the pixel traversal order with one of the default methods
        :param mode: mode of traversal
        :param order: optional user-defined traversal order
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
                sensitivity = self.get_sensitivity(transformation=lambda a: 0)
                self.traversal_orders[mode] = np.argsort(sensitivity)

            if mode == 'sensitivity_reversal':
                sensitivity = self.get_sensitivity(transformation=lambda a: 1 - a)
                self.traversal_orders[mode] = np.argsort(sensitivity)

            elif mode == 'random':
                self.traversal_orders[mode] = np.arange(self.num_pixels)
                random.shuffle(self.traversal_orders[mode])

            else:
                raise Exception('Invalid mode')

    def generate_explanation(self, epsilon, plot_path=None, verbosity=False):
        pass

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
