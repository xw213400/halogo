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

def main(step):
    records = []
    with open('../data/record.json') as json_data:
        records = json.load(json_data)

    record = records[step]

    size = math.floor(math.sqrt(len(record['board']))) - 1

    sys.setrecursionlimit(500000)
    go.init(size)
    engine = mcts.MCTSPlayerMixin(10)
    if os.path.isfile('../data/resnet_pars.pkl'):
        resnet.halo_resnet.load_state_dict(torch.load('../data/resnet_pars.pkl'))

    print('ready!')

    go.POSITION.fromJSON(record)

    print('next:', go.POSITION.next)

    move = engine.suggest_move()

    print(move)
    print(engine.debug_info)
    

if __name__ == '__main__':
    step = 0

    if len(sys.argv) >= 2:
        step = int(sys.argv[1]) - 1

    main(step)
