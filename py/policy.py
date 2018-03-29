
import go
import random


def get(position):
    positions = go.get_positions(position)
    
    for pos in positions:
        pos.prior = random.random()

    pos = go.POSITION_POOL.pop()
    position.move2(pos, 0)
    pos.prior = -1
    pos.vertex = 0
    positions.append(pos)

    sorted(positions, key=lambda pos:pos.prior)
    
    return positions

def sim():
    best_pos = go.POSITION_POOL.pop()
    go.SIM_POS.move2(best_pos, 0)
    best_pos.prior = -1
    best_pos.vertex = 0

    i = 0
    pos = go.POSITION_POOL.pop()
    while i < go.LN:
        v = go.COORDS[i]
        if go.SIM_POS.resonable(v) and go.SIM_POS.move2(pos, v):
            pos.prior = random.random()
            if pos.prior > best_pos.prior:
                tmp_pos = best_pos
                best_pos = pos
                best_pos.vertex = v
                best_pos.prior = pos.prior
                pos = tmp_pos
        i += 1
    
    go.POSITION_POOL.append(pos)
    go.POSITION_POOL.append(go.SIM_POS)
    go.SIM_POS = best_pos
