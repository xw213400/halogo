import copy
import math
import random
import sys
import time

import gtp
import numpy as np

import go
import utils
import policy

# Draw moves from policy net until this threshold, then play moves randomly.
# This speeds up the simulation, and it also provides a logical cutoff
# for which moves to include for reinforcement learning.
POLICY_CUTOFF_DEPTH = int(go.N * go.N * 0.75) # 270 moves for a 19x19
# However, some situations end up as "dead, but only with correct play".
# Random play can destroy the subtlety of these situations, so we'll play out
# a bunch more moves from a smart network before playing out random moves.
POLICY_FINISH_MOVES = int(go.N * go.N * 0.2) # 72 moves for a 19x19

def sorted_moves(probability_array):
    coords = [(a, b) for a in range(go.N) for b in range(go.N)]
    coords.sort(key=lambda c: probability_array[c], reverse=True)
    return coords

def select_random():
    policy.run()
    selection = random.random()
    cumprob = 0
    i = -1
    probs = policy.PROBS
    inv = policy.INV_SUM
    while cumprob < selection:
        i += 1
        cumprob += probs[i] * inv
    return policy.MOVES[i]

def simulate_game():
    """Simulates a game starting from a position, using a policy network"""
    max_step = len(go.HISTORY)
    pass_num = 0
    while go.STEP < max_step:
        vertex = select_random()
        if vertex == 0:
            pass_num += 1
        if pass_num >= 2:
            break
        go.move(go.NEXT, vertex, False)

class RandomPlayerMixin:
    def suggest_move(self):
        return select_random()


# All terminology here (Q, U, N, p_UCT) uses the same notation as in the
# AlphaGo paper.
# Exploration constant
c_PUCT = 1

class MCTSNode():
    '''
    A MCTSNode has two states: plain, and expanded.
    An plain MCTSNode merely knows its Q + U values, so that a decision
    can be made about which MCTS node to expand during the selection phase.
    When expanded, a MCTSNode also knows the actual position at that node,
    as well as followup moves/probabilities via the policy network.
    Each of these followup moves is instantiated as a plain MCTSNode.
    '''
    @staticmethod
    def root_node():
        node = MCTSNode(None, None, 0)
        node.step = go.STEP
        return node

    def __init__(self, parent, move, prior):
        self.parent = parent # pointer to another MCTSNode
        self.move = move # the move that led to this node
        self.prior = prior
        self.step = go.STEP# if parent is None else parent.step + 1 # lazily computed upon expansion
        self.probs = None
        self.children = {} # map of moves to resulting MCTSNode

        self.Q = 0 if self.parent is None else self.parent.Q # average of all outcomes involving this node
        self.U = prior # monte carlo exploration bonus
        self.N = 0 # number of times node was visited

    def __repr__(self):
        return "<MCTSNode move=%s prior=%s score=%s is_expanded=%s>" % (self.move, self.prior, self.action_score, self.is_expanded())

    @property
    def action_score(self):
        # Note to self: after adding value network, must calculate 
        # self.Q = weighted_average(avg(values), avg(rollouts)),
        # as opposed to avg(map(weighted_average, values, rollouts))
        return self.Q + self.U

    def expand(self):
        if self.probs is None:
            self.probs = policy.get_probs()

        if len(self.probs) > 0:
            most_prob_move = max(self.probs, key=lambda k:self.probs[k])
            node = MCTSNode(self, most_prob_move, self.probs[most_prob_move])
            self.children[most_prob_move] = node
            del self.probs[most_prob_move]

            return node

        return MCTSNode(self, 0, 0)
        
    def backup_value(self, value):
        self.N += 1
        if self.parent is None:
            # No point in updating Q / U values for root, since they are
            # used to decide between children nodes.
            return
        # This incrementally calculates node.Q = average(Q of children),
        # given the newest Q value and the previous average of N-1 values.
        global c_PUCT
        self.Q = self.Q + (value - self.Q) / self.N
        self.U = c_PUCT * math.sqrt(self.parent.N) * self.prior / self.N
        # must invert, because alternate layers have opposite desires
        self.parent.backup_value(-value)
        # in here:
        # action_score = q + (value - q) / n + C * sqrt(parent_n) * prior / n
        # in pure mcts:
        # action_score = win_n / n + C * sqrt(log(parent_n) / n)

    def select_leaf(self):
        if len(current.children) > 0:
            best_node = max(current.children.values(), key=lambda node:node.action_score)
            if best_node.action_score > none_score: #exploitation or exploration
                return best_node
        return None

SIM_COUNT = 0

class MCTSPlayerMixin:
    def __init__(self, seconds_per_move=5):
        self.seconds_per_move = seconds_per_move
        self.max_rollout_depth = len(go.HISTORY)
        self.current_node = None
        self.last_node = None
        super().__init__()

    def suggest_move(self):
        global SIM_COUNT
        SIM_COUNT = 0

        ######################################################
        root_node = MCTSNode.root_node()
        start = time.time()
        while time.time() - start < self.seconds_per_move:
            last_node = None
            current_node = root_node
            #selection
            while current_node is not None:
                last_node = current_node
                current_node = Selection(current_node)
            
            #expand
            last_node = last_node.expand()

            #simulate
            R = last_node.simulate()

            #backpropagate
            current_node = last_node
            while current_node is not None:
                backpropagation(current_node, R)
                current_node = current_node.parent

        best_move = max(root_node.children.keys(), key=lambda move, root=root_node: root_node.children[move].N, reverse=True)
        ######################################################

        self.current = self.selection(self.current)
        # print("Searched for %s seconds" % (time.time() - start), file=sys.stderr)
        print ("simulate_game_count: " + str(SIM_COUNT))
        for move, node in root.children.items():
            print (move, node.action_score, node.N)
        sorted_moves = sorted(root.children.keys(), key=lambda move, root=root: root.children[move].N, reverse=True)
        for move in sorted_moves:
            if go.is_move_reasonable(move):
                return move
        return 0

    def selection(self, node):
        # print("tree search", file=sys.stderr)
        # selection
        chosen_leaf = node.select_leaf()
        # expansion
        legal = go.move(go.NEXT, chosen_leaf.move)
        # print("Investigating following position:\n%s" % (chosen_leaf.position,), file=sys.stderr)
        chosen_leaf.expand()
        # evaluation
        value = self.estimate_value(chosen_leaf)
        # backup
        # print("value: %s" % value, file=sys.stderr)
        chosen_leaf.backup_value(value)
        go.undo()
        sys.stderr.flush()

    def expansion(self):
        node = self.current.expand()
        node.simulate()

    def estimate_value(self, chosen_leaf):
        # Estimate value of position using rollout only (for now).
        # (TODO: Value network; average the value estimations from rollout + value network)
        go.copy()
        # move = chosen_leaf.move
        # current = copy.deepcopy(leaf_position)
        simulate_game()
        global SIM_COUNT
        SIM_COUNT += 1
        # print(current, file=sys.stderr)
        score = go.score()
        go.paste()

        leaf_pos = go.HISTORY[chosen_leaf.step]
        if leaf_pos.color == go.WHITE:
            score *= -1

        return score

