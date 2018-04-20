import gtp

import go
import resnet
import random
from mcts import MCTSPlayerMixin

class GtpInterface(object):
    def __init__(self):
        go.init(7)
        resnet.init()

    def set_size(self, n):
        go.init(n)
        resnet.init()
        
    def set_komi(self, komi):
        go.KOMI = komi

    def clear(self):
        go.init(go.N)

    def make_move(self, color, vertex):
        j, i = vertex
        go.MOVE_POS.copy(go.POSITION)
        legal = go.MOVE_POS.play(j, i)
        if legal:
            # print("MAKE_MOVE:", i, j)
            captures = go.get_captures(go.POSITION, go.MOVE_POS)
            go.POSITION.copy(go.MOVE_POS)
            return True, {go.toXY(v) for v in captures }
        else:
            return False, {}

    def debug(self):
        info = self.debug_info
        info += go.POSITION.text()
        return info

    def get_move(self, color):
        move = self.suggest_move()
        return go.toXY(move)

    def get_score(self):
        return go.POSITION.result()

    def save(self):
        return go.POSITION.toJSON()

    def load(self, str):
        go.POSITION.fromJSON(str)

    def suggest_move(self, position):
        raise NotImplementedError

# class RandomPlayer(RandomPlayerMixin, GtpInterface): pass
class MCTSPlayer(MCTSPlayerMixin, GtpInterface): pass

def make_gtp_instance():
    # instance = RandomPlayer()
    instance = MCTSPlayer(30)
    gtp_engine = gtp.Engine(instance)
    return gtp_engine
