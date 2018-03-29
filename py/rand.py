class RandomPlayerMixin:
    def suggest_move(self):
        move = policy.sim()
        return policy.MOVES[i]