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
    i = 0
    while pass_num < 2:
        move = engine.suggest_move()
        i += 1
        # print(i, '###########')
        # print(engine.debug_info)
        print(i, "AAA:", go.POSITION.vertex, go.POSITION.ko, move)
        go.MOVE_POS.copy(go.POSITION)
        # print("BBB:", go.POSITION.vertex, go.POSITION.ko, move)
        legal = go.MOVE_POS.move2(go.POSITION, move)
        # print("CCC:", go.POSITION.vertex, go.POSITION.ko, move)
        if not legal:
            print('Illegal move!')
        if move == 0:
            pass_num += 1
        else:
            pass_num = 0

        record = go.POSITION.toJSON()
        records += '  '
        records += record
        # print(record)

        if pass_num < 2:
            records += ',\n'
        else:
            records += '\n]'
        
    f = open('../data/record.json', 'w')
    f.write(records)
    f.close()

    

if __name__ == '__main__':
    size = 7

    if len(sys.argv) >= 2:
        size = int(sys.argv[1])

    main(size)
