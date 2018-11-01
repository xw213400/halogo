#!/usr/bin/python

import sys
import go
import tensorflow as tf
import numpy as np
from os.path import isfile, join
from os import listdir
import json
import math
import random

INPUT_BOARD = tf.zeros([1, 1, go.N, go.N])

def input_board(position):
    global INPUT_BOARD

    c = 0
    x = 0
    y = 0
    while c < go.LN:
        v = go.COORDS[c]

        if v == position.ko:
            INPUT_BOARD[0, 0, y, x] = 2
        else:
            INPUT_BOARD[0, 0, y, x] = position.board[v] * position.next

        x += 1
        if x == go.N:
            y += 1
            x = 0
        c = y * go.N + x

def Residual_block(input, input_channel, output_channel):
    conv = tf.nn.conv2d(input, [5, 5, input_channel, output_channel], [1, 1, 1, 1], 'SAME')
    out = tf.add(conv, input)
    return tf.nn.relu(out)

def resnet(input, num_planes=32):
    conv0 = tf.nn.conv2d(input, [5, 5, 1, num_planes], [1, 1, 1, 1], 'SAME')
    conv1 = Residual_block(conv0, num_planes, num_planes)
    conv2 = Residual_block(conv1, num_planes, num_planes)
    conv3 = Residual_block(conv2, num_planes, num_planes)
    conv4 = Residual_block(conv3, num_planes, num_planes)
    conv5 = Residual_block(conv4, num_planes, num_planes)
    conv6 = tf.nn.conv2d(conv5, [1, 1, num_planes, 4], [1,1,1,1], 'VALID')
    linear = tf.reshape(conv6, [-1])
    weight = tf.Variable(tf.truncated_normal([go.LN*4, go.LN+1]), 0.25)
    return tf.matmul(linear, weight)

def loss(logits, labels):
    labels = tf.to_int64(labels)
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

def train(positions, epoch=1):
    sess = tf.Session()
    init = tf.global_variables_initializer()#初始化变量
    sess.run(init)

    # tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(0.001)

    for e in range(epoch):
        batch = 10
        n = math.floor(len(positions)/batch)
        i = 0
        running_loss = 0.0
        random.shuffle(positions)
        while i < n:
            j = 0
            target_data = tf.constant(batch)
            input_data = tf.placeholder(tf.float32, [batch, 1, go.N, go.N])
            while j < batch:
                k = i * batch + j
                pos = positions[k]
                v = go.LN
                if pos.vertex != 0:
                    p, q = go.toJI(pos.vertex)
                    v = q * go.N + p - go.N - 1

                target_data[j] = v

                input_board(pos.parent)
                input_data[j].copy_(INPUT_BOARD[0])

                j += 1

            # optimizer.zero_grad()

            # x = Variable(input_data)
            # t = Variable(target_data)
            # if torch.cuda.is_available():
            #     x = x.cuda()
            #     t = t.cuda()

            # y = resnet(x)

            y = resnet(input_data)

            loss = tf.losses.softmax_cross_entropy(y, target_data)
            loss.backward()
            optimizer.step()

            i += 1
            running_loss += loss.data[0]
            if i % 100 == 0 or i == n:
                print('epoch: %d, i:%d, loss %.3f' % (e, i*batch, running_loss / 100))
                running_loss = 0.0


def main(path, epoch=1):
    sys.setrecursionlimit(500000)
    go.init(9)
    policy = resnet(32)

    positions = []
    records = [f for f in listdir(path) if f[-4:] == 'json']
    for f in records:
        with open(path+f) as json_data:
            record = json.load(json_data)
            s = 0
            parent = go.Position()
            while s < len(record) and s <= go.LN:
                position = go.Position()
                position.fromJSON(record[s])
                position.parent = parent
                parent = position
                if position.vertex != 0:
                    positions.append(position)
                s += 1

    policy.train(positions, epoch)
    # torch.save(policy.resnet.state_dict(), path+'resnet_pars.pkl')


    # x_vals = np.random.normal(1, 0.1, 100)
    # y_vals = np.repeat(10.0, 100)

    # x_data = tf.placeholder(shape=[1], dtype=tf.float32)
    # y_target = tf.placeholder(shape=[1], dtype= tf.float32)

    # A = tf.Variable(tf.random_normal(shape=[1]))

    # my_output = tf.multiply(x_data, A)

    # loss = tf.square(my_output - y_target)

    # sess = tf.Session()
    # init = tf.global_variables_initializer()#初始化变量
    # sess.run(init)

    # my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
    # train_step = my_opt.minimize(loss)

    # for i in range(100):#0到100,不包括100 
    #     # 随机从样本中取值 
    #     rand_index = np.random.choice(100) 
    #     rand_x = [x_vals[rand_index]] 
    #     rand_y = [y_vals[rand_index]] 
    #     #损失函数引用的placeholder(直接或间接用的都算), x_data使用样本rand_x， y_target用样本rand_y 
    #     sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y}) 
    #     #打印 
    #     if i%5==0: 
    #         print('step: ' + str(i) + ' A = ' + str(sess.run(A))) 
    #         print('loss: ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))

    

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
