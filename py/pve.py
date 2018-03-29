#!/usr/bin/python

import sys
import gtp
from gtp_wrapper import make_gtp_instance

def main():
    sys.setrecursionlimit(500000)
    engine = make_gtp_instance()
    print('ready!')
    while True:
        cmd = sys.stdin.readline()
        if cmd:
            engine_reply = engine.send(cmd)

            message_id, command, arguments = gtp.parse_message(cmd)
            if command == 'play':
                c, vertex = gtp.parse_move(arguments)
                msg = 'genmove '
                if c == gtp.BLACK:
                    msg += gtp.gtp_color(gtp.WHITE)
                elif c == gtp.WHITE:
                    msg += gtp.gtp_color(gtp.BLACK)

                print(engine.send(msg))
            
            print(engine.send('debug'))

if __name__ == '__main__':
    main()
