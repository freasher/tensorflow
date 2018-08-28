from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 下载mnist库，并导入到项目中
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None,784])

w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
