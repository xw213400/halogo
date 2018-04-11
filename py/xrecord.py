#!/usr/bin/python

import sys
import go
import resnet
import mcts
import time
import torch
import os.path

def main(size):
    sys.setrecursionlimit(500000)
    go.init(size)
    engine = mcts.MCTSPlayerMixin(10)
    if os.path.isfile('../data/resnet_pars.pkl'):
        resnet.halo_resnet.load_state_dict(torch.load('../data/resnet_pars.pkl'))
    print('ready!')
    records = '[\n'
    pass_num = 0
    while pass_num < 2:
        move = engine.suggest_move()
        go.SIM_POS.copy(go.POSITION)
        legal = go.SIM_POS.move2(go.POSITION, move)
        if not legal:
            print('Illegal move!')
        if move == 0:
            pass_num += 1
        else:
            pass_num = 0

        records += '  '
        records += go.POSITION.toJSON()

        if pass_num < 2:
            records += ',\n'
        else:
            records += ']\n'
        
    f = open('../data/record.json', 'w')
    f.write(records)
    f.close()

    

if __name__ == '__main__':
    size = 7

    if len(sys.argv) >= 2:
        size = int(sys.argv[1])

    main(size)
