#!/usr/bin/python

import sys
import go
import tensorflow as tf
import numpy as np
from os.path import isfile, join
from os import listdir
import json


def main(path, epoch=1):
    x_vals = np.random.normal(1, 0.1, 100)
    y_vals = np.repeat(10.0, 100)

    x_data = tf.placeholder(shape=[1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[1], dtype= tf.float32)

    A = tf.Variable(tf.random_normal(shape=[1]))

    my_output = tf.multiply(x_data, A)

    loss = tf.square(my_output - y_target)

    sess = tf.Session()
    init = tf.global_variables_initializer()#初始化变量
    sess.run(init)

    my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
    train_step = my_opt.minimize(loss)

    for i in range(100):#0到100,不包括100 
        # 随机从样本中取值 
        rand_index = np.random.choice(100) 
        rand_x = [x_vals[rand_index]] 
        rand_y = [y_vals[rand_index]] 
        #损失函数引用的placeholder(直接或间接用的都算), x_data使用样本rand_x， y_target用样本rand_y 
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y}) 
        #打印 
        if i%5==0: 
            print('step: ' + str(i) + ' A = ' + str(sess.run(A))) 
            print('loss: ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))

    

if __name__ == '__main__':
    path = '../data/'
    epoch = 1
    if len(sys.argv) >= 2:
        path += sys.argv[1] + '/'
    if len(sys.argv) >= 3:
        epoch = int(sys.argv[2])

    if path != '../data/':
        main(path, epoch)
    else:
        print('Data is not exist!')
