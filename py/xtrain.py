#!/usr/bin/python

import sys
import go
import resnet
import mcts
import time
import torch
import os.path

def main(size, n):
    sys.setrecursionlimit(500000)
    go.init(size)
    engine = mcts.MCTSPlayerMixin(10)
    if os.path.isfile('../data/resnet_pars.pkl'):
        resnet.halo_resnet.load_state_dict(torch.load('../data/resnet_pars.pkl'))
    print('ready!')
    i = 0
    records = ''
    while i < n:
        record = ''
        go.clear()
        engine.clear()
        print(i)
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
            if pass_num < 2:
                record += '%d,' % move
            else:
                record += '0\n'
            # print(record)
        i+=1
        records += record
        torch.save(resnet.halo_resnet.state_dict(), '../data/resnet_pars.pkl')
    
    daytime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    filename = '../data/%d_%d_%s' % (size, n, daytime)
    f = open(filename, 'w')
    f.write(records)
    f.close()

    

if __name__ == '__main__':
    n = 1
    size = 7

    if len(sys.argv) >= 2:
        n = int(sys.argv[1])

    if len(sys.argv) >= 3:
        size = int(sys.argv[2])

    main(size, n)
