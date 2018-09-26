import go
from mcts import MCTSPlayer

class Engine():
    def __init__(self, time=5, policy=None):
        self.player = MCTSPlayer(time, policy)
        
    def set_komi(self, komi):
        go.KOMI = komi

    def set_size(self, size):
        go.init(size)

    def clear(self):
        self.player.clear()

    def debug(self):
        print(self.player.debug_info)

    def move(self, color, vertex=None):
        legal = True

        if vertex is None:
            vertex = self.player.move()
            legal = vertex is not None
        else:
            legal = go.move(vertex)

        if legal:
            captures = go.get_captures(go.POSITION)
            return go.toXY(go.POSITION.vertex), {go.toXY(v) for v in captures}
        else:
            return None, {}

    def get_score(self):
        return go.POSITION.result()

    def save(self):
        return go.POSITION.toJSON()

    def load(self, str):
        go.POSITION.fromJSON(str)

