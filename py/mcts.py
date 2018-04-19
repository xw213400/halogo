import time
import math
import resnet
import go

c_PUCT = go.N

class MCTSNode():
    def __init__(self, parent, position):
        self.parent = parent # pointer to another MCTSNode
        self.position = position # the move that led to this node

        self.positions = resnet.get(position) # list of Move resonable, sort by prior, PASS is always at first

        self.children = [] # map of moves to resulting MCTSNode

        self.Q = 0 if parent is None else parent.Q # average of all outcomes involving this node
        self.U = 0 #move.prior# monte carlo exploration bonus
        self.N = 0 # number of times node was visited
        self.action_score = 0

    def select(self):
        if len(self.positions) > 0 or len(self.children) == 0:
            return None
        else:
            return max(self.children, key=lambda node:node.action_score)

    def expand(self):
        if len(self.positions) > 0:
            pos = self.positions.pop()
            node = MCTSNode(self, pos)
            self.children.append(node)
            return node
        else:
            return self

    def simulate(self):
        pos = self.position
        if len(self.positions) > 0:
            pos = self.positions[-1] #last node is best node, PASS is always at first

        if pos.hash_code in go.HASH_SIM:
            return go.HASH_SIM[pos.hash_code]
        
        go.BRANCH_SIM.clear()
        go.SIM_POS.copy(pos)
        while go.SIM_POS.pass_num < 2:
            go.BRANCH_SIM.add(resnet.sim())

        score = go.SIM_POS.score()
        go.HASH_SIM[go.SIM_POS.hash_code] = score

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

    def release(self, recursive=True):
        go.POSITION_POOL.append(self.position)

        while len(self.positions) > 0:
            go.POSITION_POOL.append(self.positions.pop())
        
        if recursive:
            for child in self.children:
                child.release()


class MCTSPlayerMixin:
    def __init__(self, seconds_per_move=5):
        self.seconds_per_move = seconds_per_move
        self.debug_info = ""
        self.best_node = None
        super().__init__()

    def clear(self):
        if self.best_node is not None:
            self.best_node.release()
            self.best_node = None

    def suggest_move(self):
        root_node = None
        # print("==============================")
        if self.best_node is not None:
            for node in self.best_node.children:
                if node.position.vertex == go.POSITION.vertex:
                    root_node = node
                    # print("V:", go.POSITION.vertex, node.position.vertex)
                else:
                    node.release()
            self.best_node.release(False)
            self.best_node = None

        if root_node is None:
            pos = go.POSITION_POOL.pop()
            pos.copy(go.POSITION)
            root_node = MCTSNode(None, pos)

        # print(root_node.position.text())

        start = time.time()
        while time.time() - start < self.seconds_per_move:
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
        if len(root_node.children) > 0:
            self.best_node = max(root_node.children, key=lambda node:node.N)

            sim_count = 0
            for node in root_node.children:
                sim_count += node.N
                # self.debug_info += 'V:%d; N:%d; SCORE:%f\n' % (node.position.vertex, node.N, node.action_score)
                # self.debug_info += '[%d:%d],' % (node.position.vertex, node.N)
                if node != self.best_node:
                    node.release()
                # j, i = go.toXY(node.position.vertex)

            root_node.release(False)
            # self.debug_info += 'MOVE:%d; SIM_COUNT:%d; POOL_LEFT:%d\n' % (self.best_node.position.vertex, sim_count, len(go.POSITION_POOL))
            # root_node.release()

            resnet.train(go.POSITION, self.best_node.position.vertex)

            return self.best_node.position.vertex
        else:
            root_node.release()
            return 0

