import pandas as pd
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


x_data = np.float32(np.random.rand(2, 100))  # 随机输入
y1_data = np.dot([0.100, 0.200], x_data) + 0.300
y2_data = np.dot([0.500, 0.900], x_data) + 3.000

b1 = tf.Variable(tf.zeros([1]))
W1 = tf.Variable(tf.random.uniform([1, 2], -1.0, 1.0))
y1 = tf.matmul(W1, x_data) + b1

b2 = tf.Variable(tf.zeros([1]))
W2 = tf.Variable(tf.random.uniform([1, 2], -1.0, 1.0))
y2 = tf.matmul(W2, x_data) + b2

loss1 = tf.reduce_mean(tf.square(y1 - y1_data))
loss2 = tf.reduce_mean(tf.square(y2 - y2_data))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5) #梯度下降
train1 = optimizer.minimize(loss1)
train2 = optimizer.minimize(loss2)

# 联合训练
loss = loss1 + loss2
# 构建优化器
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化全局变量
init = tf.compat.v1.global_variables_initializer()

# 启动图
with tf.compat.v1.Session() as sess:
    sess.run(init)
    for step in range(1, 300):
        sess.run(train)
        print(step, 'W1,b1,W2,b2:', sess.run(W1), sess.run(b1), sess.run(W2), sess.run(b2))


