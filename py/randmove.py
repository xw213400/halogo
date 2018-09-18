
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

        pos = go.POSITION_POOL.pop()
        position.move2(pos, 0)
        pos.prior = 0
        positions.append(pos)

        positions = sorted(positions, key=lambda pos: pos.prior)

        return positions

    def sim(self):
        positions = go.get_positions(go.SIM_POS)

        # n = len(positions)
        # i = random.randint(0, n)

        # if i == n:
        #     go.SIM_POS.move2(go.SIM_POS, 0)
        # else:
        #     go.SIM_POS.copy(positions[i])

        n = len(positions)
        if n >= 1:
            i = random.randint(0, n-1)
            go.SIM_POS.copy(positions[i])
        else:
            go.SIM_POS.move2(go.SIM_POS, 0)


        i = 0
        while i < n:
            go.POSITION_POOL.append(positions[i])
            i += 1

        return go.SIM_POS.hash_code

