#!/usr/bin/python

import sys
import gtp
from engine import Engine
import resnet
import torch

def main():
    sys.setrecursionlimit(500000)
    engine = gtp.Engine(Engine(30, resnet.Policy()))
    while True:
        cmd = sys.stdin.readline()
        if cmd:
            engine_reply = engine.send(cmd)
            print(engine_reply)
            sys.stdout.flush()

if __name__ == '__main__':
    main()
