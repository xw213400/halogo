#!/usr/bin/python

import sys
import go
import resnet
import torch
from os.path import isfile, join
from os import listdir
import json

def main(data, epoch=1):
    sys.setrecursionlimit(500000)
    go.init(9)
    policy = resnet.Policy(1, data+'resnet_pars.pkl')

    positions = []
    records = [f for f in listdir(data) if f[-4:] == 'json']
    for f in records:
        with open(data+f) as json_data:
            record = json.load(json_data)
            s = 0
            parent = go.Position()
            while s < len(record) and s < 81:
                position = go.Position()
                position.fromJSON(record[s])
                position.parent = parent
                parent = position
                if position.vertex != 0:
                    positions.append(position)
                s += 1

    policy.train(positions, epoch)
    torch.save(policy.resnet.state_dict(), data+'resnet_pars.pkl')
    

if __name__ == '__main__':
    data = ''
    epoch = 1
    if len(sys.argv) >= 2:
        data = sys.argv[1]
    if len(sys.argv) >= 3:
        epoch = int(sys.argv[2])

    if data != '':
        main('../data/'+data+'/', epoch)
    else:
        print('Data is not exist!')
