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
    kernel = tf.random_uniform([5, 5, input_channel, output_channel], minval=-0.2, maxval=0.2)
    conv = tf.nn.conv2d(board, kernel, [
                        1, 1, 1, 1], 'SAME')
    out = tf.add(conv, board)
    return tf.nn.relu(out)


def train(trainset, evalset, epoch=1):
    board = tf.placeholder(tf.float32, [None, go.N, go.N, 1])
    labels = tf.placeholder(tf.int64, [None])

    num_planes = 32

    kernel = tf.random_uniform([5, 5, 1, num_planes], minval=-0.2, maxval=0.2)
    conv0 = tf.nn.conv2d(board, kernel, [1, 1, 1, 1], 'SAME')
    conv1 = Residual_block(conv0, num_planes, num_planes)
    conv2 = Residual_block(conv1, num_planes, num_planes)
    conv3 = Residual_block(conv2, num_planes, num_planes)
    conv4 = Residual_block(conv3, num_planes, num_planes)
    conv5 = Residual_block(conv4, num_planes, num_planes)
    kernel2 = tf.random_uniform([1, 1, num_planes, 4], minval=-1./num_planes, maxval=1./num_planes)
    conv6 = tf.nn.conv2d(conv5, kernel2, [1, 1, 1, 1], 'VALID')
    linear = tf.reshape(conv6, [-1, go.LN * 4])
    weight = tf.Variable(tf.random_uniform([go.LN * 4, go.LN + 1], minval=-0.25/go.LN, maxval=0.25/go.LN))
    predict = tf.matmul(linear, weight)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=predict)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    opt = tf.train.GradientDescentOptimizer(0.001)
    train_step = opt.minimize(cross_entropy_mean)

    # iii = 0

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(max_to_keep=2)

        for e in range(epoch):
            batch = 10
            n = math.floor(len(trainset) / batch)
            i = 0
            loss = 0

            random.shuffle(trainset)
            while i < n:
                j = 0
                y_data = np.zeros(batch, dtype=np.float32)
                x_data = np.zeros(
                    batch * go.LN, dtype=np.float32).reshape(batch, go.N, go.N, 1)
                while j < batch:
                    k = i * batch + j
                    pos = trainset[k]
                    v = go.LN
                    if pos.vertex != 0:
                        p, q = go.toJI(pos.vertex)
                        v = q * go.N + p - go.N - 1

                    y_data[j] = v
                    input_board(pos.parent, x_data[j])

                    j += 1

                # iii += 1
                # if iii == 20:
                #     print(sess.run(predict, feed_dict={board: x_data}).shape)

                sess.run(train_step, feed_dict={labels: y_data, board: x_data})
                loss += sess.run(cross_entropy_mean, feed_dict={labels: y_data, board: x_data})

                i += 1

            loss /= n

            right = 0
            for pos in evalset:
                if pos.vertex != 0:
                    x_data = np.zeros(go.LN, dtype=np.float32).reshape(1, go.N, go.N, 1)
                    p, q = go.toJI(pos.vertex)
                    v = q * go.N + p - go.N - 1
                    input_board(pos.parent, x_data[0])
                    prediction = sess.run(predict, feed_dict={board: x_data})
                    sortedmoves = np.argsort(prediction[0])[::-1]
                    if v == sortedmoves[0]:
                        right += 1

            # ratio = right/len(evalset)*100.0

            print("epoch: %d, loss: %f, right: %d / %d" % (e, loss, right, len(evalset)))

            saver.save(sess, './module/goai_tf', global_step=e+1)



def main(epoch=1):
    sys.setrecursionlimit(500000)
    go.init(9)

    trainset = []
    evalset = []
    trainfiles = [f for f in listdir('../data/train/') if f[-4:] == 'json']
    testfiles = [f for f in listdir('../data/test/') if f[-4:] == 'json']
    for f in trainfiles:
        with open('../data/train/' + f) as json_data:
            record = json.load(json_data)
            s = 0
            parent = go.Position()
            while s < len(record) and s <= go.LN * 0.75:
                position = go.Position()
                position.fromJSON(record[s])
                position.parent = parent
                parent = position
                if position.vertex != 0:
                    trainset.append(position)
                s += 1

    for f in testfiles:
        with open('../data/test/' + f) as json_data:
            record = json.load(json_data)
            s = 0
            parent = go.Position()
            while s < len(record) and s <= go.LN * 0.75:
                position = go.Position()
                position.fromJSON(record[s])
                position.parent = parent
                parent = position
                if position.vertex != 0:
                    evalset.append(position)
                s += 1

    train(trainset, evalset, epoch)


if __name__ == '__main__':
    epoch = 1
    if len(sys.argv) >= 2:
        epoch = int(sys.argv[1])

    main(epoch)
