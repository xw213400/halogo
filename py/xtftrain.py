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


def input_board(position, board):

    c = 0
    x = 0
    y = 0
    while c < go.LN:
        v = go.COORDS[c]

        if v == position.ko:
            board[y, x, 0] = 2.0
        else:
            board[y, x, 0] = position.board[v] * position.next * 1.0

        x += 1
        if x == go.N:
            y += 1
            x = 0
        c = y * go.N + x


def Residual_block(board, input_channel, output_channel):
    kernel = tf.random_normal([5, 5, input_channel, output_channel])
    conv = tf.nn.conv2d(board, kernel, [
                        1, 1, 1, 1], 'SAME')
    out = tf.add(conv, board)
    return tf.nn.relu(out)

def train(positions, epoch=1):
    board = tf.placeholder(tf.float32, [None, go.N, go.N, 1])
    labels = tf.placeholder(tf.float32, [None, go.LN+1])

    num_planes = 32

    kernel = tf.random_normal([5, 5, 1, num_planes])
    conv0 = tf.nn.conv2d(board, kernel, [1, 1, 1, 1], 'SAME')
    conv1 = Residual_block(conv0, num_planes, num_planes)
    conv2 = Residual_block(conv1, num_planes, num_planes)
    conv3 = Residual_block(conv2, num_planes, num_planes)
    conv4 = Residual_block(conv3, num_planes, num_planes)
    conv5 = Residual_block(conv4, num_planes, num_planes)
    kernel2 = tf.random_normal([1, 1, num_planes, 4])
    conv6 = tf.nn.conv2d(conv5, kernel2, [1, 1, 1, 1], 'VALID')
    linear = tf.reshape(conv6, [-1, go.LN*4])
    weight = tf.Variable(tf.random_normal([go.LN*4, go.LN+1]))
    predict = tf.matmul(linear, weight)

    softmax = tf.nn.softmax(predict)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels *
                                                  tf.log(softmax+0.0000000001), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(
        0.01).minimize(cross_entropy)
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for e in range(epoch):
        batch = 10
        n = math.floor(len(positions)/batch)
        i = 0

        random.shuffle(positions)
        while i < n:
            j = 0
            y_data = np.zeros(
                batch*(go.LN+1), dtype=np.float32).reshape(batch, go.LN+1)
            x_data = np.zeros(
                batch*go.LN, dtype=np.float32).reshape(batch, go.N, go.N, 1)
            while j < batch:
                k = i * batch + j
                pos = positions[k]
                v = go.LN
                if pos.vertex != 0:
                    p, q = go.toJI(pos.vertex)
                    v = q * go.N + p - go.N - 1

                y_data[j][v] = 1
                input_board(pos.parent, x_data[j])

                j += 1

            sess.run(train_step, feed_dict={labels: y_data, board: x_data})
            print(sess.run(softmax, feed_dict={board: x_data}))

            i += 1


def main(path, epoch=1):
    sys.setrecursionlimit(500000)
    go.init(9)

    positions = []
    records = [f for f in listdir(path) if f[-4:] == 'json']
    for f in records:
        with open(path+f) as json_data:
            record = json.load(json_data)
            s = 0
            parent = go.Position()
            while s < len(record) and s <= go.LN * 0.75:
                position = go.Position()
                position.fromJSON(record[s])
                position.parent = parent
                parent = position
                if position.vertex != 0:
                    positions.append(position)
                s += 1

    train(positions, epoch)


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
