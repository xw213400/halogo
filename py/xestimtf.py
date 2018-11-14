import tensorflow as tf
import os
import sys
from tensorflow.python.platform import gfile
import numpy as np
from scipy.misc import imread, imresize
from os.path import isfile, join
from os import listdir
import go
import json

sys.setrecursionlimit(500000)
go.init(9)

def input_board(position, board):
    c = 0
    x = 0
    y = 0
    while c < go.LN:
        v = go.COORDS[c]

        if v == position.ko:
            board[y, x] = 2.0
        else:
            board[y, x] = position.board[v] * position.next * 1.0

        x += 1
        if x == go.N:
            y += 1
            x = 0
        c = y * go.N + x

#tensorboard --logdir='/home/xie/Documents/code/halogo/py/log'
with tf.Session() as sess:
    with gfile.FastGFile("../data/goai.pb", 'rb') as f:
        graph = tf.get_default_graph()
        graphdef = graph.as_graph_def()
        graphdef.ParseFromString(f.read())
        _ = tf.import_graph_def(graphdef)
        summary_write = tf.summary.FileWriter("./log", graph)
        predict = graph.get_tensor_by_name('import/add_7:0')

    evalset = []
    testfiles = [f for f in listdir('../data/estimate/') if f[-4:] == 'json']

    for f in testfiles:
        with open('../data/estimate/' + f) as json_data:
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

    right = 0
    for pos in evalset:
        if pos.vertex != 0:
            x_data = np.zeros(go.LN, dtype=np.float32).reshape(1, 1, go.N, go.N)
            p, q = go.toJI(pos.vertex)
            v = q * go.N + p - go.N - 1
            input_board(pos.parent, x_data[0][0])
            prediction = sess.run(predict, feed_dict={'import/0:0': x_data})
            sortedmoves = np.argsort(prediction[0])[::-1]
            if v == sortedmoves[0]:
                right += 1
    print("EST: %d | %d" % (right, len(evalset)))
