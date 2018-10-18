#!/usr/bin/python

import sys
import go
import torch
import os.path
from os import listdir
import randmove
import resnet
from mcts import MCTSPlayer

def main(count, path):
    sys.setrecursionlimit(500000)
    go.init(9)
    playerBlack = MCTSPlayer(30, resnet.Policy(40, '../data/r1/resnet_pars.pkl'))
    playerWhite = MCTSPlayer(30, randmove.Policy(40))

    records = [f for f in listdir(path) if f[-4:] == 'json' and f[:6] == 'record']
    fcount = len(records)

    vertex = None
    caps = None

    black_win = 0
    white_win = 0
    draw = 0
    
    c = 1
    while c <= count:
        records = '[\n'
        i = 0

        print('ready: %d in %d, POSPOOL: %d' % (c, count, len(go.POSITION_POOL)))

        while go.POSITION.pass_count() < 2:
            i += 1

            legal = True
            if i % 2 == 1:
                legal = playerBlack.move()
            else:
                legal = playerWhite.move()

            if not legal:
                print('Illegal move!')
                break

            go.POSITION.debug()

            record = go.POSITION.toJSON()
            records += '  '
            records += record

            if go.POSITION.pass_count() < 2:
                records += ',\n'
            else:
                records += '\n]'
        
        score = go.POSITION.score() - go.KOMI
        playerBlack.clear()
        playerWhite.clear()
        go.clear()

        fcount += 1
        filename = path + 'record_%d.json' % fcount

        f = open(filename, 'w')
        f.write(records)
        f.close()

        if score > 0:
            black_win += 1
        elif score < 0:
            white_win += 1
        else:
            draw += 1
        
        c += 1

    print('back win: %d, white win: %d, draw: %d' % (black_win, white_win, draw))

    

if __name__ == '__main__':
    count = 1
    path = '../data/'

    if len(sys.argv) >= 2:
        path += sys.argv[1] + '/'

    if len(sys.argv) >= 3:
        count = int(sys.argv[2])

    main(count, path)
