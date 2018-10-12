import time
import math
import resnet
import randmove
import go

POLICY = None
WORST_SCORE = -1000000000

class MCTSNode():
    def __init__(self, parent, position):
        self.parent = parent # pointer to another MCTSNode
        self.position = position # the move that led to this node

        if position.pass_count() == 2:
            self.positions = []
        else:
            self.positions = POLICY.get(position) # list of Move resonable, sort by prior, PASS is always at first

        self.leaves = len(self.positions)

        if self.parent is not None:
            self.parent.add_leaf(self.leaves) 

        self.children = [] # map of moves to resulting MCTSNode

        self.Q = 0 #if parent is None else parent.Q # average of all outcomes involving this node
        self.U = 0 # monte carlo exploration bonus
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
            best_score = WORST_SCORE
            best_node = None
            for node in self.children:
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
            #last node is best node, PASS is always at first
            return POLICY.sim(self.positions[-1])
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
        if self.position not in go.TRUNK:     
            go.POSITION_POOL.append(self.position)

        for p in self.positions:
            go.POSITION_POOL.append(p)
        
        if recursive:
            for child in self.children:
                child.release()


class MCTSPlayer():
    def __init__(self, seconds_per_move=5, policy=None):
        self.seconds_per_move = seconds_per_move
        self.best_node = None
        self.debug_info = ''

        if policy == None:
            self.policy = resnet.Policy()
        else:
            self.policy = policy

    def clear(self):
        self.policy.clear()
        if self.best_node is not None:
            self.best_node.release()
            self.best_node = None

    def move(self):
        global POLICY
        POLICY = self.policy
        root_node = None

        if self.best_node is not None:
            if self.best_node.position.next == go.POSITION.next:
                if self.best_node.position.vertex == go.POSITION.vertex:
                    root_node = self.best_node
                    self.best_node = None
            else:
                for node in self.best_node.children:
                    if node.position.vertex == go.POSITION.vertex:
                        root_node = node
                    else:
                        node.release()
                self.best_node.release(False)
                self.best_node = None

        if root_node is None:
            root_node = MCTSNode(None, go.POSITION)
        elif root_node.position is not go.POSITION:
            go.POSITION_POOL.append(root_node.position)
            root_node.position = go.POSITION
            for node in root_node.children:
                node.position.parent = go.POSITION
            for pos in root_node.positions:
                pos.parent = go.POSITION

        start = time.time()
        sim_count = 0
        while time.time() - start < self.seconds_per_move:
            #selection
            current_node = root_node.select()

            #if every child node is exploid, find the best directly
            if current_node is None:
                print("Leaves is empty!")
                break
            
            #expand
            current_node = current_node.expand()

            #simulate
            R = current_node.simulate()

            #backpropagate
            while current_node is not None:
                current_node.backpropagation(R)
                current_node = current_node.parent

            sim_count += 1
            if sim_count >= 10000:
                break

        if len(root_node.children) > 0:
            poolsize = len(go.POSITION_POOL)
            self.best_node = max(root_node.children, key=lambda node:node.N)

            vertex = self.best_node.position.vertex
            go.move(vertex)

            self.debug_info = '%02d  ' % go.get_step()
            for node in root_node.children:
                # self.debug_info += '%d,' % node.N
                if node != self.best_node:
                    node.release()

            root_node.release(False)

            i, j = go.toJI(vertex)
            self.debug_info += 'V:[%d,%d]  POOL:%d  Q:%.1f  SIM:%d' % (i, j, poolsize, self.best_node.Q, sim_count)

            return vertex
        else:
            print('!!!!!!')
            root_node.release()
            return None

