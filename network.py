from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Reshape, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
import tensorflow as tf
import tensorflow_ranking as tfr
import tf2onnx


# # train GTSRB model
# # save as gtsrb.h5 and gtsrb.onnx
#
# import pickle
# with open('GTSRB/train.p', 'rb') as f:
#     gtsrb_train = pickle.load(f)
# with open('GTSRB/valid.p', 'rb') as f:
#     gtsrb_valid = pickle.load(f)
# with open('GTSRB/test.p', 'rb') as f:
#     gtsrb_test = pickle.load(f)
#
# x_train, y_train = gtsrb_train['features'], gtsrb_train['labels']
# x_valid, y_valid = gtsrb_valid['features'], gtsrb_valid['labels']
# x_test, y_test = gtsrb_test['features'], gtsrb_test['labels']
#
#
#
# labels = [1, 2, 3, 4, 5, 7, 8]
#
#
# import numpy as np
# def mask_classes(x, y, labels):
#     mask = np.zeros_like(y)
#     for i in labels:
#         mask = np.logical_or(mask, y == i)
#     x = x[mask]
#     y = y[mask]
#     return x, y
#
# def report_counts(labels, y):
#     for i in labels:
#         count = y[y == i]
#         print("class :", i, "count : ", count.shape[0])
#
# def fix_labels(labels):
#     labels = labels-1
#     for i in range(labels.shape[0]):
#         if labels[i] > 5:
#             labels[i] = labels[i] - 1
#     return labels
#
#
# x_train, y_train = mask_classes(x_train, y_train, labels)
# x_valid, y_valid = mask_classes(x_valid, y_valid, labels)
# x_test, y_test = mask_classes(x_test, y_test, labels)
#
# y_train = fix_labels(y_train)
# y_valid = fix_labels(y_valid)
# y_test = fix_labels(y_test)
#
# # gtsrb = dict()
# # gtsrb['x_train'] = x_train
# # gtsrb['y_train'] = y_train
# # gtsrb['x_valid'] = x_valid
# # gtsrb['y_valid'] = y_valid
# # gtsrb['x_test'] = x_test
# # gtsrb['y_test'] = y_test
# # with open('GTSRB/gtsrb.pickle', 'wb') as handle:
# #     pickle.dump(gtsrb, handle)
# # with open('GTSRB/gtsrb.pickle', 'rb') as handle:
# #     gtsrb = pickle.load(handle)
#
#
# x_train = x_train/255
# x_valid = x_valid/255
# x_test = x_test/255
#
# from keras.utils.np_utils import to_categorical
# y_train = to_categorical(y_train)
# y_valid = to_categorical(y_valid)
# y_test = to_categorical(y_test)
#
#
# model_name = 'gtsrb-30-20'
#
# model = Sequential(name=model_name)
# model.add(Flatten(input_shape=(32, 32, 3), name='input'))
# model.add(Dense(30, name='dense_1'))
# model.add(Activation('relu', name='relu_1'))
# # model.add(Dropout(0.2))
# model.add(Dense(20, name='dense_2'))
# model.add(Activation('relu', name='relu_2'))
# # model.add(Dropout(0.2))
# # model.add(Dense(200, name='dense_3'))
# # model.add(Activation('relu', name='relu_3'))
# # model.add(Dropout(0.2))
# model.add(Dense(7, name='logit'))
# model.add(Activation('softmax', name='output'))
# model.summary()
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=keras.optimizers.Adam(learning_rate=0.001),
#               metrics=['accuracy'])
#
# from keras.preprocessing.image import ImageDataGenerator
# datagen = ImageDataGenerator()
# model.fit(datagen.flow(x=x_train, y=y_train, batch_size=64),
#           steps_per_epoch=100,
#           epochs=30,
#           validation_data=(x_valid, y_valid),
#           shuffle=1)
# score = model.evaluate(x_test, y_test, verbose=1)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])
#
# directory = 'models/'
# model_path = directory + model_name + '.h5'
# model_proto, _ = tf2onnx.convert.from_keras(model, output_path=directory + model_name + '.onnx')
# model.save(model_path)





# train MNIST model
# save as mnist.h5 and mnist.onnx

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# x_train /= 255
# x_test /= 255

# model = Sequential(name='mnist-10x3')
# # model.add(Reshape(([784]), input_shape=(28, 28, 1)))
# model.add(Flatten(name='input'))
# model.add(Dense(10, name='dense_1'))
# model.add(Activation('relu', name='relu_1'))
# model.add(Dense(10, name='dense_2'))
# model.add(Activation('relu', name='relu_2'))
# model.add(Dense(10, name='logit'))
# # model.add(Activation('softmax', name='output'))

# model_name = 'mnist-10x2'
# model = Sequential(name=model_name)
# model.add(Flatten(name='input'))
# model.add(Dense(10, name='dense_1', activation='relu'))
# # model.add(Activation('relu', name='relu_1'))
# model.add(Dense(10, name='dense_2', activation='relu'))
# # model.add(Activation('relu', name='relu_2'))
# model.add(Dense(10, name='logit'))
# # model.add(Activation('softmax', name='output'))

model_name = 'mnist-30x2'
model = Sequential(name=model_name)
model.add(Flatten(name='input'))
model.add(Dense(30, name='dense_1', activation='relu'))
# model.add(Activation('relu', name='relu_1'))
model.add(Dense(30, name='dense_2', activation='relu'))
# model.add(Activation('relu', name='relu_2'))
model.add(Dense(10, name='logit'))

# model_name = 'mnist-cnn'
# model = Sequential(name=model_name)
# model.add(Conv2D(4, (3, 3), name='conv_1', input_shape=(28, 28, 1)))
# model.add(Conv2D(4, (3, 3), name='conv_2'))
# model.add(Flatten())
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, name='logit'))

# model_name = 'mnist-sota'
# model = Sequential(name=model_name)
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(200, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(10, name='logit'))


# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(200, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(200, activation='relu'))
# # model.add(Dense(10, activation='softmax'))
# model.add(Dense(10, name='logit'))
# model.add(Activation('softmax', name='output'))

model.compile(# loss='categorical_crossentropy',
              # loss=tf.keras.losses.CategoricalCrossentropy(),
              loss=tfr.keras.losses.SoftmaxLoss(),
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

model.summary()
# model.save('mnist-10x2.h5')
model.save(model_name + '.h5')
model_proto, _ = tf2onnx.convert.from_keras(model, output_path=model_name + '.onnx')




# from matplotlib import pyplot as plt
# def plot_figure(image, path, cmap=None):
#     fig = plt.figure()
#     # ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax = plt.Axes(fig, [-0.5, -0.5, 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)
#     if cmap is None:
#         plt.imshow(image)
#     else:
#         plt.imshow(image, cmap=cmap)
#     plt.savefig(path, bbox_inches='tight')
#     plt.close(fig)
#
# import numpy as np
# from keras import backend
# from keras.models import load_model
# from keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# # y_test = tf.keras.utils.to_categorical(y_test, 10)
# x_test = x_test.astype('float32') / 255
#
# index = 0
# image = x_test[index]
# model = load_model('mnist.h5')
# label = model.predict(np.expand_dims(image, axis=0)).argmax()
#
# temp = x_test[index].reshape(28*28)
# image_batch = np.kron(np.ones((28*28, 1)), temp)
# image_batch_manipulated = image_batch.copy()
# for i in range(28*28):
#     # image_batch_manipulated[i][i] = 1 - image_batch_manipulated[i][i]
#     image_batch_manipulated[i][i] = 0
# func = backend.function([model.layers[0].input],
#                         [model.layers[model.layers.__len__() - 1].input])
# predictions = func([image_batch.reshape(784, 28, 28, 1)])[0]
# predictions_manipulated = func([image_batch_manipulated.reshape(784, 28, 28, 1)])[0]
# difference = predictions - predictions_manipulated
# sensitivity = difference[:, label]
# plot_figure(sensitivity.reshape(28, 28), 'x.png')
