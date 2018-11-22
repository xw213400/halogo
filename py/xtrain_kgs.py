#!/usr/bin/python

import sys
import go
import resnet
import torch
from os.path import isfile, join
from os import listdir
import json
import sgf

SGF_COLUMNS = 'abcdefghijklmnopqrs'
    

def replay_sgf(sgf_contents):
    collection = sgf.parse(sgf_contents)
    game = collection.children[0]

    handicap = 0
    HA = game.root.properties.get('HA')
    if HA is not None:
        handicap = int(HA[0])

    if handicap == 0:
        position = go.POSITION_POOL.pop()
        node = game.root
        while node is not None:
            s = ''

            if 'B' in node.properties:
                s = node.properties.get('B', [''])[0]
            elif 'W' in node.properties:
                s = node.properties.get('W', [''])[0]

            vertex = 0
            if len(s) == 2:
                j = SGF_COLUMNS.index(s[0])
                i = SGF_COLUMNS.index(s[1])
                vertex = (i+1) * go.M + j + 1

            pos = position.move(vertex)

            if pos is None:
                return None
                # position.debug()
                # print(s + ": %d, %d, %d" % (i, j, vertex))

            pos.update_group()
            position = pos

            node = node.next
        
        return position

    return None


def main(path, epoch=1):
    sys.setrecursionlimit(500000)
    policy = resnet.Policy(1, path+'goai.pth')

    trainset = []
    trainpos = [f for f in listdir(path+'train') if f[-4:] == '.sgf']
    print("========")
    for f in trainpos:
        with open(path+'train/'+f) as sf:
            position = replay_sgf(sf.read())
            while position is not None and position.parent is not None:
                trainset.append(position)
                position = position.parent

    estimset = []
    estimpos = [f for f in listdir(path+'estimate') if f[-4:] == '.sgf']
    print("========")
    for f in estimpos:
        with open(path+'estimate/'+f) as sf:
            position = replay_sgf(sf.read())
            while position is not None and position.parent is not None:
                estimset.append(position)
                position = position.parent

    policy.train(trainset, estimset, epoch)

    # save cpp version module
    # policy.resnet.save(path+'goai.pt')
    

if __name__ == '__main__':
    epoch = 1
    if len(sys.argv) >= 2:
        epoch = int(sys.argv[1])

    main('../kgs-19-2017-08/', epoch)
