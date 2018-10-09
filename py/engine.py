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

    def debug(self, info=''):
        print(self.player.debug_info + info)

    def move(self, color, vertex=None):
        legal = True

        if vertex is None:
            vertex = self.player.move()
            legal = vertex is not None
        else:
            legal = go.move(vertex)

        if legal:
            take = go.get_take(go.POSITION)
            return go.toJI(go.POSITION.vertex), {go.toJI(v) for v in take}
        else:
            return None, {}

    def get_score(self):
        return go.POSITION.result()

    def save(self):
        return go.POSITION.toJSON()

    def load(self, str):
        go.POSITION.fromJSON(str)

