#!/usr/bin/python

import sys
import gtp
from gtp_wrapper import make_gtp_instance
import resnet

def main():
    sys.setrecursionlimit(500000)
    engine = make_gtp_instance()
    if os.path.isfile('../data/resnet_pars.pkl'):
        resnet.halo_resnet.load_state_dict(torch.load('../data/resnet_pars.pkl'))
    while True:
        cmd = sys.stdin.readline()
        if cmd:
            engine_reply = engine.send(cmd)
            print(engine_reply)
            sys.stdout.flush()

if __name__ == '__main__':
    main()
