#!/usr/bin/python

import sys
import go
import resnet
import torch
from os.path import isfile, join
from os import listdir
import json

def main(path, epoch=1):
    sys.setrecursionlimit(500000)
    go.init(9)
    policy = resnet.Policy(1, path+'resnet_pars.pkl')

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

    policy.train(positions, epoch)
    torch.save(policy.resnet.state_dict(), path+'resnet_pars.pkl')
    

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
