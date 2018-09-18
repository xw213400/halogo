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
    engineB = Engine(20, Policy(30))
    engineW = Engine(20, Policy(20))

    move = None
    legal = None
    caps = None

    black_win = 0
    white_win = 0
    draw = 0
    
    c = 1
    while c <= count:
        go.clear()
        engineB.start()
        engineW.start()

        records = '[\n'
        pass_num = 0
        i = 0

        print('ready: %d in %d' % (c, count))

        while pass_num < 2:
            i += 1

            if i % 2 == 1:
                move = engineB.get_move('b')
                legal, caps = engineB.make_move('b', move)
            else:
                move = engineW.get_move('w')
                legal, caps = engineW.make_move('w', move)
            
            print(i, move, caps)

            if not legal:
                print('Illegal move!')
                break

            if move == (0, 0):
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
