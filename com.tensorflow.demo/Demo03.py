import tensorflow as tf


# 构造阶段
matirix1 = tf.constant([[3.,3.]])

matirix2 = tf.constant([[2.],[2.]])

product = tf.matmul(matirix1,matirix2)

# 启动会话
sess = tf.Session()
# 执行会话
result = sess.run(product)
print(result)
# 关闭会话
sess.close()

# with tf.Session as sess:
#     result = sess.run(product)
#     print result