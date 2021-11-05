import tensorflow as tf
import numpy as np
import pandas as pd

tf.compat.v1.disable_eager_execution()

#数据预处理
f = open('49_EXL.csv', encoding='utf-8')
c1 = f.readlines()
head = c1[4:6]
h1 = head[0].split(',')
h2 = head[1].split(',')
for i in range(len(h1)):
    if h1[i] == '':
        h1[i] = h2[i]
header = h1

data_set = pd.read_csv('49.csv', header=None, names=[i for i in header])

#训练集和测试集的划分
train_set = pd.read_csv('49.csv', nrows = 800, header=None, names=[i for i in header])
test_set = data_set.drop(i for i in range(800))




#神经网络
x_data = [[i] for i in train_set['AGE']]
y_data = [[j] for j in train_set['PMV']]

print(x_data)

x_te = [[i] for i in test_set['AGE']]
y_te = [[j] for j in test_set['PMV']]

# x = tf.compat.v1.placeholder(tf.float32, [None, 1])
# y = tf.compat.v1.placeholder(tf.float32, [None, 1])


# Weights_L1 = tf.Variable(tf.compat.v1.random_normal([1, 10]))
# biases_L1 = tf.Variable(tf.zeros([1, 10]))
# Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
# L1 = tf.nn.tanh(Wx_plus_b_L1)
#
# Weights_L2 = tf.Variable(tf.compat.v1.random_normal([10, 1]))
# biases_L2 = tf.Variable(tf.zeros([1, 1]))
# Wx_plus_b_L2 = tf.matmul(L1, Weights_L2)+biases_L2
# prediction = tf.nn.tanh(Wx_plus_b_L2)

W = tf.Variable(tf.compat.v1.random_normal([1, 10]))
b = tf.Variable(tf.zeros([1, 10]))
prediction = W*x_data+b

# loss = tf.reduce_mean(tf.square(y-prediction))
# train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


loss = tf.reduce_mean(tf.square(y_data-prediction))
#使用梯度下降法
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.2).minimize(loss)

#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y_data, 1), tf.argmax(prediction, 1))
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(21):
        for step in range(2000):
            sess.run(train_step, feed_dict={x_data: x_data, prediction: y_data})
        acc = sess.run(accuracy, feed_dict={x_te: x_te, prediction: y_te})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))