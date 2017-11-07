from tqdm import tqdm
import numpy as np
import os
import tflearn
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from random import shuffle
from preprocess import create_train_data, process_test_data
import cv2
import random
# import tensorflow as tf
import keras
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
# K.set_image_dim_ordering('th')
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
# import tflearn
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.estimator import regression
# from tflearn.layers.conv import conv_2d, max_pool_2d

CURR_DIR = os.getcwd()

TRAIN_DIR = CURR_DIR + '/train'
TEST_DIR = CURR_DIR + '/test'
IMG_SIZE = 50
LR = 1e-3

# BATCH_SIZE = 128

MODEL_NAME = 'dogvscats-{}--{}.model'.format(LR,'6convbasic_keras')


test_data = np.load("test_data.npy")
train_data = np.load("train_data.npy")


def conv_nn_model():
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(50,50,1)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation='softmax'))
	return model


def train_model():

	model = conv_nn_model()
	Y =[]
	test_y=[]

	# if os.path.exists('{}.meta'.format(MODEL_NAME)):
	# 	model.load(MODEL_NAME)
	# 	print("Model loaded!")


	train = train_data[:-500]
	test = train_data[500:]

	X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

	for i in train:
		if np.array_equal(i[1],np.array([1,0])):
			Y.append(0)
		elif np.array_equal(i[1],np.array([0,1])):
			Y.append(1)


	Y = keras.utils.to_categorical(Y,2)

	# Y = [i[1] for i in train]

	test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

	for i in test:
		if np.array_equal(i[1],np.array([1,0])):
			test_y.append(0)
		elif np.array_equal(i[1],np.array([0,1])):
			test_y.append(1)


	test_y = keras.utils.to_categorical(test_y,2)

	# test_y = [i[1] for i in test]
	model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

	model.fit(X, Y,
          epochs=5,
          verbose=1,
          validation_data=(test_x, test_y),)

    


def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(tf.Graph.as_graph_def(), 'out', \
        MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
        False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")


	# model.fit(X,Y, n_epoch=7, validation_set=(test_x,test_y),
	# 		 snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

	# model.save(MODEL_NAME)


# train_model()
model = conv_nn_model()
train_model()
# model.load(MODEL_NAME)

export_model(tf.train.Saver(),model,["conv2d_1_input"], "dense_2/Softmax")

# img = test_data[random.randint(1,12500)]

# prediction = np.argmax(model.predict(img[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1)))

# if prediction == 0:
# 	pred = 'cat'
# elif prediction == 1:
# 	pred = 'dog'

# print(pred)

# img_path = os.path.join(TEST_DIR,img[1] + '.jpg')

# cv2.imshow('image',cv2.imread(img_path))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# def tensorboard_stuff():

# 	tensorboard --logdir=/Users/arnav1712/code/cats_vs_dogs/log














