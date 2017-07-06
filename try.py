# coding:utf-8
import tensorflow as tf
import numpy as np


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('input_layer'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')


with tf.name_scope('hidden_layer'):
    with tf.name_scope('weight'):
        W1 = tf.Variable(tf.random_normal([1, 10]))
        tf.summary.histogram('hidden_layer/weight', W1)
    with tf.name_scope('bias'):
        b1 = tf.Variable(tf.zeros([1, 10]) + 0.1)
        tf.summary.histogram('hidden_layer/bias', b1)
    with tf.name_scope('Wx_plus_b'):
        Wx_plus_b1 = tf.matmul(xs, W1) + b1
        tf.summary.histogram('hidden_layer/Wx_plus_b', Wx_plus_b1)
output1 = tf.nn.relu(Wx_plus_b1)


with tf.name_scope('output_layer'):
    with tf.name_scope('weight'):
        W2 = tf.Variable(tf.random_normal([10, 1]))
        tf.summary.histogram('output_layer/weight', W2)
    with tf.name_scope('bias'):
        b2 = tf.Variable(tf.zeros([1, 1]) + 0.1)
        tf.summary.histogram('output_layer/bias', b2)
    with tf.name_scope('Wx_plus_b'):
        Wx_plus_b2 = tf.matmul(output1, W2) + b2
        tf.summary.histogram('output_layer/Wx_plus_b', Wx_plus_b2)
output2 = Wx_plus_b2

# 损失
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - output2), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs', sess.graph)

# 训练
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if (i % 50 == 0):
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)
