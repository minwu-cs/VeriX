from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Reshape, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
import tensorflow as tf
# import tensorflow_ranking as tfr
import tf2onnx


# train GTSRB model
# save as gtsrb.h5 and gtsrb.onnx

import pickle
with open('GTSRB/train.p', 'rb') as f:
    gtsrb_train = pickle.load(f)
with open('GTSRB/valid.p', 'rb') as f:
    gtsrb_valid = pickle.load(f)
with open('GTSRB/test.p', 'rb') as f:
    gtsrb_test = pickle.load(f)

x_train, y_train = gtsrb_train['features'], gtsrb_train['labels']
x_valid, y_valid = gtsrb_valid['features'], gtsrb_valid['labels']
x_test, y_test = gtsrb_test['features'], gtsrb_test['labels']



labels = [2,1,13,12,38,10,4,5,25,9]
labelToIndex = {}
for i, l in enumerate(labels):
    labelToIndex[l] = i

import numpy as np
def mask_classes(x, y, labels):
    mask = np.zeros_like(y)
    for i in labels:
        mask = np.logical_or(mask, y == i)
    x = x[mask]
    y = y[mask]
    return x, y

def report_counts(labels, y):
    for i in labels:
        count = y[y == i]
        print("class :", i, "count : ", count.shape[0])

def fix_labels(labels):
    for i in range(labels.shape[0]):
        labels[i] = labelToIndex[labels[i]]
    return labels

x_train, y_train = mask_classes(x_train, y_train, labels)
x_valid, y_valid = mask_classes(x_valid, y_valid, labels)
x_test, y_test = mask_classes(x_test, y_test, labels)

y_train = fix_labels(y_train)
y_valid = fix_labels(y_valid)
y_test = fix_labels(y_test)

gtsrb = dict()
gtsrb['x_train'] = x_train
gtsrb['y_train'] = y_train
gtsrb['x_valid'] = x_valid
gtsrb['y_valid'] = y_valid
gtsrb['x_test'] = x_test
gtsrb['y_test'] = y_test
with open('GTSRB/gtsrb.pickle', 'wb') as handle:
    pickle.dump(gtsrb, handle)
# with open('GTSRB/gtsrb.pickle', 'rb') as handle:
#     gtsrb = pickle.load(handle)


x_train = x_train/255
x_valid = x_valid/255
x_test = x_test/255

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)
y_test = to_categorical(y_test)


# model_name = 'gtsrb-30-20'
# model = Sequential(name=model_name)
# model.add(Flatten(input_shape=(32, 32, 3), name='input'))
# model.add(Dense(30, name='dense_1'))
# model.add(Activation('relu', name='relu_1'))
# model.add(Dense(20, name='dense_2'))
# model.add(Activation('relu', name='relu_2'))
# model.add(Dropout(0.2))
# model.add(Dense(10, name='logit'))
# model.add(Activation('softmax', name='output'))

# model_name = 'gtsrb-10x2'
# model = Sequential(name=model_name)
# model.add(Flatten(input_shape=(32, 32, 3), name='input'))
# model.add(Dense(10, name='dense_1', activation='relu'))
# model.add(Dense(10, name='dense_2', activation='relu'))
# model.add(Dense(10, name='logit'))
# model.add(Activation('softmax', name='output'))
#
# model_name = 'gtsrb-30x2'
# model = Sequential(name=model_name)
# model.add(Flatten(input_shape=(32, 32, 3), name='input'))
# model.add(Dense(30, name='dense_1', activation='relu'))
# model.add(Dense(30, name='dense_2', activation='relu'))
# model.add(Dense(10, name='logit'))
# model.add(Activation('softmax', name='output'))
#
# model_name = 'gtsrb-cnn'
# model = Sequential(name=model_name)
# model.add(Conv2D(4, (3, 3), name='conv_1', input_shape=(32, 32, 3)))
# model.add(Conv2D(4, (3, 3), name='conv_2'))
# model.add(Flatten())
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, name='logit'))
# model.add(Activation('softmax', name='output'))

model_name = 'gtsrb-sota-small'
model = Sequential(name=model_name)
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, name='logit'))
model.add(Activation('softmax', name='output'))

# model_name = 'gtsrb-sota'
# model = Sequential(name=model_name)
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
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
# model.add(Activation('softmax', name='output'))

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()
model.fit(datagen.flow(x=x_train, y=y_train, batch_size=64),
          steps_per_epoch=100,
          epochs=50,
          validation_data=(x_valid, y_valid),
          shuffle=1)
score = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

directory = 'models/'
model_path = directory + model_name + '.h5'
model_proto, _ = tf2onnx.convert.from_keras(model, output_path=directory + model_name + '.onnx')
model.save(model_path)



"""

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

model = Sequential(name='mnist-10x2')
# model.add(Reshape(([784]), input_shape=(28, 28, 1)))
model.add(Flatten(name='input'))
model.add(Dense(10, name='dense_1'))
model.add(Activation('relu', name='relu_1'))
model.add(Dense(10, name='dense_2'))
model.add(Activation('relu', name='relu_2'))
model.add(Dense(10, name='logit'))
# model.add(Activation('softmax', name='output'))

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
# model.add(Dense(10, activation='softmax'))

model.compile(loss=tfr.keras.losses.SoftmaxLoss(),
              # loss='categorical_crossentropy',
              # loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=32,
          epochs=20,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

model.summary()
# model.save('mnist-10x2.h5')
model.save('x.h5')
model_proto, _ = tf2onnx.convert.from_keras(model, output_path='x.onnx')

"""


