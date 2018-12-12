#!/usr/bin/python

import sys
import go
import resnet2
import torch
from os.path import isfile, join
from os import listdir
import json
import math

def main(path, epoch=1):
    sys.setrecursionlimit(500000)
    policy = resnet2.Policy(1, path+'goai.pth')

    maxN = math.ceil(go.LN * 0.8)

    trainset = []
    trainpos = [f for f in listdir(path+'train') if f[-4:] == 'json']
    for f in trainpos:
        with open(path+'train/'+f) as json_data:
            record = json.load(json_data)
            s = 0
            parent = go.Position()
            while s < len(record) and s <= maxN:
                position = go.Position()
                position.fromJSON(record[s])
                position.parent = parent
                parent = position
                if position.vertex != 0:
                    trainset.append(position)
                s += 1

    estimset = []
    estimpos = [f for f in listdir(path+'estimate') if f[-4:] == 'json']
    for f in estimpos:
        with open(path+'estimate/'+f) as json_data:
            record = json.load(json_data)
            s = 0
            parent = go.Position()
            while s < len(record) and s <= maxN:
                position = go.Position()
                position.fromJSON(record[s])
                position.parent = parent
                parent = position
                if position.vertex != 0:
                    estimset.append(position)
                s += 1

    policy.train(trainset, estimset, epoch)

    # save cpp version module
    # policy.resnet.save(path+'goai.pt')
    

if __name__ == '__main__':
    epoch = 1
    if len(sys.argv) >= 2:
        epoch = int(sys.argv[1])

    main('../data-9-rand/', epoch)
