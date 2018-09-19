import go
from mcts import MCTSPlayer

class Engine():
    def __init__(self, time=5, policy=None):
        self.player = MCTSPlayer(time, policy)
        
    def set_komi(self, komi):
        go.KOMI = komi

    def set_size(self, size):
        go.init(size)

    def start(self):
        self.player.clear()

    def debug(self):
        info = self.player.debug_info
        info += go.POSITION.text()
        return info

    def move(self, color, vertex=None):
        if vertex is None:
            legal = self.player.move()
            if legal:
                captures = go.get_captures(go.POSITION)
                return go.toXY(go.POSITION.vertex), {go.toXY(v) for v in captures}
            else:
                return None, {}
        else:
            pos = go.POSITION.move(vertex)
            if pos is not None:
                go.POSITION = pos
                captures = go.get_captures(go.POSITION)
                return go.toXY(vertex), {go.toXY(v) for v in captures}
            else:
                return None, {}

    def get_score(self):
        return go.POSITION.result()

    def save(self):
        return go.POSITION.toJSON()

    def load(self, str):
        go.POSITION.fromJSON(str)

