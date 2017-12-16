import os
import urllib.request
import argparse
import sys
import alexnet
import cv2
import tensorflow as tf
import numpy as np
import caffe_classes

parser = argparse.ArgumentParser(prog='test1.py', description='default script')
parser.add_argument('--path', dest='path', help='the image path', metavar='PATH', required=True)
args = parser.parse_args(sys.argv[1:])

img = cv2.imread(args.path)

dropoutPro = 1
classNum = 1000
skipLayers = []

IMG_MEAN = np.array([104,117,124], np.float)
x = tf.placeholder('float', [1,227,227,3])

model = alexnet.AlexNet(x, dropoutPro, classNum, skipLayers)
score = model.fc8
softmax = tf.nn.softmax(score)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	model.load_initial_weights(sess)
	res = cv2.resize(img.astype(np.float), (227, 227)) - IMG_MEAN
	ans = np.argmax(sess.run(softmax, feed_dict = {x: res.reshape(1, 227, 227, 3)}))
	ans = caffe_classes.class_names[ans]
	print(ans)
