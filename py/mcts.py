import time
import math
import resnet
import randmove
import go

POLICY = None

class MCTSNode():
    def __init__(self, parent, position):
        self.parent = parent # pointer to another MCTSNode
        self.position = position # the move that led to this node

        if position.pass_count() == 2:
            self.positions = []
        else:
            self.positions = POLICY.get(position) # list of Move resonable, sort by prior, PASS is always at first

        self.leaves = len(self.positions)
        self.PN = self.leaves

        if self.parent is not None:
            self.parent.add_leaf(self.leaves) 

        self.children = [] # map of moves to resulting MCTSNode

        self.Q = 0 if parent is None else parent.Q # average of all outcomes involving this node
        self.U = 0 #move.prior# monte carlo exploration bonus
        self.N = 0 # number of times node was visited
        self.action_score = 0

    def add_leaf(self, leaves):
        self.leaves += leaves
        if self.parent is not None:
            self.parent.add_leaf(leaves)

    def sub_leaf(self):
        self.leaves -= 1
        if self.parent is not None:
            self.parent.sub_leaf()

    def select(self):
        if len(self.positions) > 0:
            return self
        else:
            n = len(self.children)
            i = 0
            best_score = go.WORST_SCORE
            best_node = None
            while i < n:
                node = self.children[i]
                i += 1
                if node.leaves > 0 and node.action_score > best_score:
                    best_score = node.action_score
                    best_node = node

            if best_node is None:
                return None
            else:
                return best_node.select()

    def expand(self):
        pos = self.positions.pop()
        node = MCTSNode(self, pos)
        self.children.append(node)
        self.sub_leaf()

        return node

    def simulate(self):
        if len(self.positions) > 0:
            pos = sim_root = self.positions[-1] #last node is best node, PASS is always at first

            while pos.pass_count() < 2:
                pos = POLICY.sim(pos)

            score = pos.score()
            while pos is not sim_root:
                go.POSITION_POOL.append(pos)
                pos = pos.parent

            return score
        else: # 2 pass
            return self.position.score()
        
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
        
        self.Q = self.Q + (score - self.Q) / self.N
        self.U = POLICY.PUCT * math.sqrt(math.log(self.parent.N + 1) / self.N)
        self.action_score = self.Q + self.U

    def release(self, recursive=True):
        if self.position.hash_code not in go.TRUNK:        
            go.POSITION_POOL.append(self.position)

        while len(self.positions) > 0:
            go.POSITION_POOL.append(self.positions.pop())
        
        if recursive:
            for child in self.children:
                child.release()


class MCTSPlayer():
    def __init__(self, seconds_per_move=5, policy=None):
        self.seconds_per_move = seconds_per_move
        self.debug_info = ""
        self.best_node = None

        if policy == None:
            self.policy = resnet.Policy()
        else:
            self.policy = policy

    def clear(self):
        if self.best_node is not None:
            self.best_node.release()
            self.best_node = None

    def move(self):
        global POLICY
        POLICY = self.policy
        root_node = None
        go.update_trunk()
        # self.debug_info = ""
        # print("==============================")
        if self.best_node is not None:
            if self.best_node.position.next == go.POSITION.next:
                if self.best_node.position.vertex == go.POSITION.vertex:
                    root_node = self.best_node
                    self.best_node = None
            else:
                for node in self.best_node.children:
                    if node.position.vertex == go.POSITION.vertex:
                        root_node = node
                        # print("V:", go.POSITION.vertex, node.position.vertex)
                    else:
                        node.release()
                self.best_node.release(False)
                self.best_node = None

        if root_node is None:
            root_node = MCTSNode(None, go.POSITION)

        # print(root_node.position.text())
        # self.debug_info += 'ROOT_LEAF:%d; ' % (root_node.leaves)

        start = time.time()
        while time.time() - start < self.seconds_per_move:
            #selection
            current_node = root_node.select()

            #if every child node is exploid, find the best directly
            if current_node is None:
                print("WARNING! ROOT has no leaves!")
                break
            
            #expand
            current_node = current_node.expand()

            #simulate
            R = current_node.simulate()

            #backpropagate
            while current_node is not None:
                current_node.backpropagation(R)
                current_node = current_node.parent

        if len(root_node.children) > 0:
            self.best_node = max(root_node.children, key=lambda node:node.N)

            vertex = self.best_node.position.vertex
            go.POSITION = go.POSITION.move(vertex)

            go.update_trunk()

            sim_count = 0
            for node in root_node.children:
                sim_count += node.N
                if node != self.best_node:
                    node.release()

            root_node.release(False)

            # self.debug_info += 'MOVE:%d; BEST_LEAF:%d; SIM_COUNT:%d\n' % (self.best_node.position.vertex, self.best_node.leaves, sim_count)

            return True
        else:
            go.update_trunk()
            root_node.release()
            return False

