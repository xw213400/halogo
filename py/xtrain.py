#!/usr/bin/python

import sys
import go
import resnet
import torch
from os.path import isfile, join
from os import listdir
import json

def main(data):
    sys.setrecursionlimit(500000)
    go.init(9)
    policy = resnet.Policy(1, data+'resnet_pars.pkl')

    positions = []
    records = [f for f in listdir(data) if f[-4:] == 'json']
    for f in records:
        with open(data+f) as json_data:
            record = json.load(json_data)
            for step in record:
                position = go.Position()
                position.fromJSON(step)
                positions.append(position)

    policy.train(positions)
    torch.save(policy.resnet.state_dict(), data+'resnet_pars.pkl')
    

if __name__ == '__main__':
    data = ''
    if len(sys.argv) >= 2:
        data = sys.argv[1]

    if data != '':
        main('../data/'+data+'/')
    else:
        print('Data is not exist!')
