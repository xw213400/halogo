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

    go.init(size)
   
    print('ready!')

    go.POSITION.fromJSON(record)

    print(go.POSITION.next)

    print(go.POSITION.text())

    go.POSITION.input_board()

    print(go.text_flag_board())

    positions = go.get_positions(go.POSITION)
    
    for p in positions:
        print(go.toXY(p.vertex))
    

if __name__ == '__main__':
    step = 0

    if len(sys.argv) >= 2:
        step = int(sys.argv[1]) - 1

    main(step)
