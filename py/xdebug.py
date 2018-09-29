#!/usr/bin/python

import sys
import go
import resnet
import mcts
import time
import torch
import os.path
import json
import math
from torch.autograd import Variable

def main(step):
    records = []
    with open('../data/record.json') as json_data:
        records = json.load(json_data)

    sys.setrecursionlimit(500000)
    go.init(9)
    net = resnet.Resnet(32)
    pars = '../data/resnet_pars.pkl'
    if os.path.isfile(pars):
        net.load_state_dict(torch.load(pars))

    parent = go.Position()
    position = go.Position()

    if step > 0:
        parent.fromJSON(records[step-1])
    position.fromJSON(records[step])

    position.parent = parent

    parent.input_board()

    x = Variable(go.INPUT_BOARD)
    out = net(x)[0].data.numpy()

    print('next:', parent.next)
    parent.debug()

    i = go.N
    s = "\n"
    while i > 0:
        i -= 1
        j = 0
        while j < go.N:
            c = out[i*go.N+j]
            j += 1
            if c < 0:
                s += '\033[32m%03d \033[0m' % int(-c*10)
            else:
                s += '\033[31m%03d \033[0m' % int(c*10)
        s += "\n\n"
    
    print(s)
    print(out[go.LN-1])


if __name__ == '__main__':
    step = 0

    if len(sys.argv) >= 2:
        step = int(sys.argv[1]) - 1

    main(step)
