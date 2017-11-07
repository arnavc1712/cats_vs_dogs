
from tqdm import tqdm
import numpy as np
import os
import tflearn
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from random import shuffle
# from preprocess import create_train_data, process_test_data
import cv2
import random
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d, max_pool_2d

CURR_DIR = os.getcwd()

TRAIN_DIR = CURR_DIR + '/train'
TEST_DIR = CURR_DIR + '/test'
IMG_SIZE = 50


def label_img(img):
	word_label = img.split('.')[-3]

	if word_label == "dog": return [0,1]
	elif word_label =="cat": return [1,0]


def create_train_data():
	training_set = []

	for img in tqdm(os.listdir(TRAIN_DIR)):
		word_label = label_img(img)
		path = os.path.join(TRAIN_DIR,img)
		img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE,IMG_SIZE))
		training_set.append(np.array(img),np.array(word_label))

	shuffle(training_set)
	np.save("train_data.npy",training_set)
	return training_set



def process_test_data():
	testing_set = []

	for img in tqdm(os.listdir(TEST_DIR)):
		img_num = img.split('.')[0]
		path = os.path.join(TEST_DIR,img)
		img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE,IMG_SIZE))
		testing_set.append([np.array(img), img_num])

	np.save("test2_data.npy",testing_set)
	return testing_set


# process_test_data()







