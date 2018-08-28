from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 下载mnist库，并导入到项目中
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None,784])

w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,w) + b)

# 我们首先需要添加一个新的占位符用于输入正确值
y_ = tf.placeholder("float", [None,10])

# 然后我们可以用  计算交叉熵:
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# TensorFlow会用你选择的优化算法来不断地修改变量以降低成本
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)