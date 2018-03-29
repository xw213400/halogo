import time
import math
import resnet
import go

c_PUCT = 15


class MCTSNode():
    def __init__(self, parent, position):
        self.parent = parent # pointer to another MCTSNode
        self.position = position # the move that led to this node

        self.positions = resnet.get(position) # list of Move resonable, sort by prior, PASS is always first
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

        if pos.hash_code in go.HASH_SIM:
            return go.HASH_SIM[pos.hash_code]

        if pos.vertex == 0:
            pass_num += 1
        else:
            pass_num = 0

        if pass_num >= 2:
            score = pos.score()
            go.HASH_SIM[go.SIM_POS.hash_code] = score
            return score
        
        codes = []
        go.SIM_POS.copy(pos)
        i = 0
        while pass_num < 2:
            codes.append(resnet.sim())
            if go.SIM_POS.vertex == 0:
                pass_num += 1
            else:
                pass_num = 0

            i += 1
            if i > go.LN * 10 and i < go.LN * 10 + 10:
                print(i, go.toXY(go.SIM_POS.vertex), go.SIM_POS.hash_code, go.SIM_POS.hash_code in go.HASH_SIM)
                print(go.SIM_POS.text())

        score = go.SIM_POS.score()

        for hc in codes:
            go.HASH_SIM[hc] = score

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
        self.best_node = max(root_node.children, key=lambda node:node.N)

        sim_count = 0
        for node in root_node.children:
            sim_count += node.N
            if node != self.best_node:
                node.release()
            # j, i = go.toXY(node.position.vertex)

        root_node.release(False)
        self.debug_info += 'MOVE:%d; SIM_COUNT:%d; POOL_LEFT:%d\n' % (self.best_node.position.vertex, sim_count, len(go.POSITION_POOL))
        # root_node.release()

        resnet.train(go.POSITION, self.best_node.position.vertex)

        return self.best_node.position.vertex
