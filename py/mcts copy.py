import time
import math
import policy
import go

c_PUCT = 15


class MCTSNode():
    def __init__(self, parent, position):
        self.parent = parent # pointer to another MCTSNode
        self.position = position # the move that led to this node

        self.positions = policy.get(position) # list of Move resonable, sort by prior, PASS is always first
        self.children = [] # map of moves to resulting MCTSNode

        self.Q = 0 if parent is None else parent.Q # average of all outcomes involving this node
        self.U = 0 #move.prior# monte carlo exploration bonus
        self.N = 0 # number of times node was visited
        self.action_score = 0

    def select(self):
        if len(self.positions) > 0:
            return None
        else:
            best_node = max(self.children, key=lambda node:node.action_score)
            return best_node

    def expand(self):
        pos = self.positions.pop()
        node = MCTSNode(self, pos)
        self.children.append(node)
        return node

    def simulate(self):
        pass_num = 0

        if self.position.vertex == 0:
            pass_num += 1

        pos = self.positions[-1]
        if pos.vertex == 0:
            pass_num += 1
        else:
            pass_num = 0

        if pass_num >= 2:
            return pos.score()

        go.SIM_POS.copy(pos)
        while pass_num < 2:
            policy.sim()
            if go.SIM_POS.vertex == 0:
                pass_num += 1
            else:
                pass_num = 0

        score = go.SIM_POS.score()

        return score
        
    def backpropagation(self, value):
        score = value
        # must invert, because alternate layers have opposite desires
        if self.position.next == go.BLACK:
            score = -value

        self.N += 1
        if self.parent is None:
            # No point in updating Q / U values for root, since they are
            # used to decide between children nodes.
            return
        # This incrementally calculates node.Q = average(Q of children),
        # given the newest Q value and the previous average of N-1 values.
        # self.Q = self.Q + (score - self.Q) / self.N
        # self.U = c_PUCT * math.sqrt(self.parent.N) * self.move.prior / self.N
        # in here:
        # action_score = q + (score - q) / n + C * sqrt(parent_n) * prior / n
        # in pure mcts:
        # action_score = win_n / n + C * sqrt(log(parent_n) / n)
        
        self.Q = self.Q + (score - self.Q) / self.N
        self.U = c_PUCT * math.sqrt(math.log(self.parent.N + 1) / self.N)
        self.action_score = self.Q + self.U

    def release(self):
        go.POSITION_POOL.append(self.position)
        self.move = None

        while len(self.positions) > 0:
            go.POSITION_POOL.append(self.positions.pop())
        
        for child in self.children:
            child.release()


class MCTSPlayerMixin:
    def __init__(self, seconds_per_move=5):
        self.seconds_per_move = seconds_per_move
        self.debug_info = ""
        super().__init__()

    def suggest_move(self):
        pos = go.POSITION_POOL.pop()
        pos.copy(go.POSITION)
        root_node = MCTSNode(None, pos)
        start = time.time()
        while time.time() - start < self.seconds_per_move:
        # while time.time() - start < 1000000:
            last_node = None
            current_node = root_node
            #selection
            while current_node is not None:
                last_node = current_node
                current_node = current_node.select()
            
            #expand
            last_node = last_node.expand()

            #simulate
            R = last_node.simulate()

            #backpropagate
            current_node = last_node
            while current_node is not None:
                current_node.backpropagation(R)
                current_node = current_node.parent

        self.debug_info = ""
        best = max(root_node.children, key=lambda node:node.N)
        sim_count = 0
        for node in root_node.children:
            sim_count += node.N
            j, i = go.toXY(node.position.vertex)
            self.debug_info += 'MOVE:%d,%d; N:%d; Score:%f\n' % (j, i, node.N, node.action_score)
        self.debug_info += 'SIM_COUNT:%d\n' % sim_count
        best_move = best.position.vertex
        self.debug_info += 'POOL_LEFT:%d\n' % len(go.POSITION_POOL)
        root_node.release()
        self.debug_info += 'POOL_SUM:%d\n' % len(go.POSITION_POOL)

        return best_move
