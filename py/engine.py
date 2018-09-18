import go
from mcts import MCTSPlayer

class Engine():
    def __init__(self, time=5, policy='resnet', pars=None):
        self.player = MCTSPlayer(time, policy, pars)
        
    def set_komi(self, komi):
        go.KOMI = komi

    def clear(self, size=0):
        if size == 0:
            size = go.N
        go.init(size)

    def make_move(self, color, vertex):
        j, i = vertex
        go.MOVE_POS.copy(go.POSITION)
        legal = go.MOVE_POS.play(j, i)
        if legal:
            captures = go.get_captures(go.POSITION, go.MOVE_POS)
            go.POSITION.copy(go.MOVE_POS)
            return True, {go.toXY(v) for v in captures }
        else:
            return False, {}

    def debug(self):
        info = self.player.debug_info
        info += go.POSITION.text()
        return info

    def get_move(self, color):
        move = self.player.suggest_move()
        return go.toXY(move)

    def get_score(self):
        return go.POSITION.result()

    def save(self):
        return go.POSITION.toJSON()

    def load(self, str):
        go.POSITION.fromJSON(str)

