#!/usr/bin/python

import sys
import go
import torch
import os.path
from engine import Engine
import randmove
import resnet

def main(count, path):
    sys.setrecursionlimit(500000)
    go.init(9)
    engineB = Engine(90, resnet.Policy(32, '../data/rand_40/resnet_pars.pkl'))
    # engineB = Engine(30, randmove.Policy(40))
    engineW = Engine(30, randmove.Policy(40))

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
        score = 0
        pass_num = 0
        i = 0

        print('ready: %d in %d, POSPOOL: %d' % (c, count, len(go.POSITION_POOL)))

        while pass_num < 2:
            i += 1

            if i % 2 == 1:
                vertex, caps = engineB.move('b')
                engineB.debug()
            else:
                vertex, caps = engineW.move('w')
                engineW.debug()

            if vertex is None:
                print('Illegal move!')
                break

            if vertex == (0, 0):
                pass_num += 1
            else:
                pass_num = 0

            record = go.POSITION.toJSON()
            records += '  '
            records += record

            if pass_num < 2 and i <= go.LN:
                records += ',\n'
            else:
                records += '\n]'
                score = go.POSITION.score()
                engineB.clear()
                engineW.clear()
                go.clear()
                print('POSPOOL: %d' % len(go.POSITION_POOL))
        
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
        count = int(sys.argv[1])

    if len(sys.argv) >= 3:
        path += sys.argv[1] + '/'

    main(count, path)
