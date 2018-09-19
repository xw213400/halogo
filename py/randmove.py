
import go
import random


class Policy():
    def __init__(self, PUCT=1):
        self.PUCT = PUCT
        pass

    def get(self, position):
        positions = go.get_positions(position)
    
        for pos in positions:
            pos.prior = random.random()

        pos = position.move(0)
        pos.prior = 0
        positions.append(pos)

        positions = sorted(positions, key=lambda pos: pos.prior)

        return positions

    def sim(self, position):
        positions = go.get_positions(position)

        n = len(positions)
        if n >= 1:
            i = random.randint(0, n-1)
            pos = positions.pop(i)
            for p in positions:
                go.POSITION_POOL.append(p)
            return pos
        else:
            return position.move(0)

