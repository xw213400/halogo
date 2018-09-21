#!/usr/bin/python

import sys
import go
import torch
import os.path
from engine import Engine
from randmove import Policy

def main(count):
    sys.setrecursionlimit(500000)
    go.init(9)
    engineB = Engine(3, Policy(25))
    engineW = Engine(3, Policy(20))

    vertex = None
    caps = None

    black_win = 0
    white_win = 0
    draw = 0
    
    c = 1
    while c <= count:
        records = '[\n'
        pass_num = 0
        i = 0

        print('ready: %d in %d, POSPOOL: %d' % (c, count, len(go.POSITION_POOL)))

        while pass_num < 2:
            i += 1

            if i % 2 == 1:
                vertex, caps = engineB.move('b')
            else:
                vertex, caps = engineW.move('w')
            
            print(i, vertex, caps)

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

            if pass_num < 2:
                records += ',\n'
            else:
                records += '\n]'
                engineB.clear()
                engineW.clear()
                go.clear()
                print('POSPOOL: %d' % len(go.POSITION_POOL))
        
        filename = '../data/record.json'
        if count > 1:
            filename = '../data/record_%d.json' % c

        f = open(filename, 'w')
        f.write(records)
        f.close()

        score = go.POSITION.score()
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

    if len(sys.argv) >= 2:
        count = int(sys.argv[1])

    main(count)
