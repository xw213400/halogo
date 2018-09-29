
import go
import random


class Policy():
    def __init__(self, PUCT=1):
        self.PUCT = PUCT
        self.HASH = {}

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
        score = self.HASH.get(position.hash_code)
        if score is not None:
            return score

        pos = position

        while pos.pass_count() < 2:
            positions = go.get_positions(pos)
            n = len(positions)
            if n >= 2:
                i = random.randint(0, n-1)
                pos = positions.pop(i)
                for p in positions:
                    go.POSITION_POOL.append(p)
            elif n == 1:
                pos = positions[0]
            else:
                pos = pos.move(0)

        score = pos.score()
        while pos is not position:
            go.POSITION_POOL.append(pos)
            pos = pos.parent
        
        self.HASH[position.hash_code] = score

        return score

    def clear(self):
        self.HASH = {}

