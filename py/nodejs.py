#!/usr/bin/python

import sys
import gtp
from gtp_wrapper import make_gtp_instance

def main():
    sys.setrecursionlimit(500000)
    engine = make_gtp_instance()
    while True:
        cmd = sys.stdin.readline()
        if cmd:
            engine_reply = engine.send(cmd)
            print(engine_reply)
            sys.stdout.flush()

if __name__ == '__main__':
    main()
