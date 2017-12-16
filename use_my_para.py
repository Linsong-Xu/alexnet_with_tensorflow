import tensorflow as tf
from alexnet import AlexNet
import matplotlib.pyplot as plt
import os
import urllib.request
import argparse
import sys

class_name = ['angry', 'hate', 'scared', 'happy', 'sad', 'surprised', '<->']

parser = argparse.ArgumentParser(prog='my_img.py', description='protect my own img script')
parser.add_argument('--path', dest='path', help='the image path', metavar='PATH', required=True)
args = parser.parse_args(sys.argv[1:])

def test_image(path_image, num_class, weights_path='Default'):
    img_string = tf.read_file(path_image)
    img_decoded = tf.image.decode_png(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [227, 227])
    img_resized = tf.reshape(img_resized, shape=[1, 227, 227, 3])
    model = AlexNet(img_resized, 0.5, 7, skip_layer='', weights_path=weights_path)
    score = tf.nn.softmax(model.fc8)
    max = tf.argmax(score, 1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './store/checkpoints/model_epoch10.ckpt')
        print(sess.run(model.fc8))
        prob = sess.run(max)[0]
        plt.imshow(img_decoded.eval())
        plt.title('Class:' + class_name[prob])
        plt.axis('off')
        plt.show()


test_image(args.path, num_class=7)
