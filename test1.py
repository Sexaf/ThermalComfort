# Author:沈栩烽
# Date:2021.1
# For Graduation Dissertation

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
tf.compat.v1.disable_eager_execution()

# 表格预处理部分
f = open('49_EXL.csv', encoding='utf-8')
c1 = f.readlines()
head = c1[4:6]
h1 = head[0].split(',')
h2 = head[1].split(',')
for i in range(len(h1)):
    if h1[i] == '':
        h1[i] = h2[i]
header = h1

c2 = c1[6:]
c2 = ''.join(c2)
f = open('49.csv', 'w')
f.write(c2)
df = pd.read_csv('49.csv', header=None, names=[i for i in header])


# 多任务学习处理数据集
x_data = [[i] for i in df['AGE']]
y_data = [[j] for j in df['PMV']]
# 定义两个placeholder
x = tf.compat.v1.placeholder(tf.float32, [None, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.compat.v1.random_normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.compat.v1.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2)+biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 二次损失函数
loss = tf.reduce_mean(tf.square(y-prediction))
# 定义一个梯度下降法来进行训练的优化器 学习率0.1
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.compat.v1.Session() as sess:
    # 变量初始化
    sess.run(tf.compat.v1.global_variables_initializer())
    # 训练2000次
    for step in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    # 获得预测值
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    # 画图展示结果
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()

