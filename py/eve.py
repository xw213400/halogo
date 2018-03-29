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
    if os.path.isfile('resnet_pars.pkl'):
        resnet.halo_resnet.load_state_dict(torch.load('resnet_pars.pkl'))
    print('ready!')
    i = 0
    record = ''
    while i < n:
        go.POSITION.copy_board(go.EMPTY_BOARD)
        go.POSITION.next = go.BLACK
        go.POSITION.ko = 0
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
            print(record)
        i+=1
        print(i)
        torch.save(resnet.halo_resnet.state_dict(), 'resnet_pars.pkl')

    daytime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    filename = './%d_%d_%s' % (size, n, daytime)
    f = open(filename, 'w')
    f.write(record)
    f.close()

    

if __name__ == '__main__':
    size = int(sys.argv[1]) or 7
    n = int(sys.argv[2]) or 1
    main(size, n)
