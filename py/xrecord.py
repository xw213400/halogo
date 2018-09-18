#!/usr/bin/python

import sys
import go
import torch
import os.path
from engine import Engine

def main(size):
    sys.setrecursionlimit(500000)
    go.init(size)
    engineB = Engine(10, 'randmove')
    engineW = Engine(10, 'randmove')
    
    print('ready!')
    records = '[\n'
    pass_num = 0
    i = 0
    move = None
    legal = None
    caps = None
    while pass_num < 2:
        i += 1

        if i % 2 == 1:
            move = engineB.get_move('b')
            legal, caps = engineB.make_move('b', move)
        else:
            move = engineW.get_move('w')
            legal, caps = engineW.make_move('w', move)
        
        print(i, move, legal, caps)

        if not legal:
            print('Illegal move!')
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
        
    f = open('../data/record.json', 'w')
    f.write(records)
    f.close()

    

if __name__ == '__main__':
    size = 9

    if len(sys.argv) >= 2:
        size = int(sys.argv[1])

    main(size)
