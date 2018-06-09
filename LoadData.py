import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class mnist_data(object):
    def __init__(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

    def load_train(self):
        x = self.mnist.train.images.reshape(-1, 28, 28, 1)
        x = tf.cast(x, tf.float32)/255.0
        with tf.Session():
            x = x.eval()
        return x, self.mnist.train.labels

    def load_test(self):
        x = self.mnist.test.images.reshape(-1, 28, 28, 1)
        x = tf.cast(x, tf.float32)
        return x, self.mnist.test.labels
