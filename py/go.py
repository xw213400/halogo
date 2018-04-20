
import torch
import random
import json
# Represent a board as a numpy array, with 0 empty, 1 is black, -1 is white.
# This means that swapping colors is as simple as multiplying array by -1.
WHITE, EMPTY, BLACK, WALL = range(-1, 3)
LEFT = -1
RIGHT = 1

# # # # #
# + + +
# + + +
# + + +
# # # #
N = 7
LN = 49
M = 8
LM = 8 * 9 + 1

UP = M
DOWN = -M

LEFTUP = LEFT + UP
LEFTDOWN = LEFT + DOWN
RIGHTUP = RIGHT + UP
RIGHTDOWN = RIGHT + DOWN

KOMI = 6.5
FLAG = 0
FLAGS = None
EMPTY_BOARD = None
COORDS = None
FRONTIER = None
POSITION = None
SIM_POS = None
MOVE_POS = None
POSITION_POOL = None

BRANCH_SIM = None # only this simulation branch positions
HASH_SIM = None # only history expanded node positions
CODE_WHITE = None
CODE_BLACK = None
CODE_KO = None
CODE_SWAP = random.getrandbits(64)

INPUT_BOARD = None
FLAG_BOARD = None #用于标记是否resonable
HASH_BOARD = None #标记该位置是否为下过的局面,用于避免循环局面，模拟开始时记录局面估值


class Position:
    def __init__(self):
        self.next = BLACK
        self.ko = 0
        self.board = EMPTY_BOARD[:]
        self.prior = 0
        self.vertex = 0
        self.hash_code = 0
        self.pass_num = 0

    # prepare input plane for resnet
    # INPUT_BOARD[0]: enemy:1, empty:2, self:3
    # INPUT_BOARD[1]: unresonable:1, resonable:2, ko:3
    def input_board(self):
        for v in COORDS:
            j, i = toXY(v)
            j -= 1
            i -= 1
            p = 1
            FLAG_BOARD[v] = False
            HASH_BOARD[v] = False
            if self.resonable(v) and self.move2(MOVE_POS, v):
                p = 2
                FLAG_BOARD[v] = True
            if v == self.ko:
                p = 3
            if MOVE_POS.hash_code in BRANCH_SIM:
                HASH_BOARD[v] = True
            INPUT_BOARD[0, 0, i, j] = self.board[v] * self.next + 2
            INPUT_BOARD[0, 1, i, j] = p

    def init_hash_code(self):
        self.hash_code = 0
        if self.next == WHITE:
            self.hash_code = self.hash_code ^ CODE_SWAP
        self.hash_code ^= CODE_KO[self.ko]
        for v in COORDS:
            if self.board[v] == BLACK:
                self.hash_code ^= CODE_BLACK[v]
            elif self.board[v] == WHITE:
                self.hash_code ^= CODE_WHITE[v]


    def resonable(self, v):
        if v == 0:
            return True

        if self.board[v] != EMPTY or v == self.ko:
            return False

        v1 = self.board[v + UP]
        v2 = self.board[v + DOWN]
        v3 = self.board[v + LEFT]
        v4 = self.board[v + RIGHT]

        if v1 * v2 * v3 * v4 == 0:
            return True

        e = -self.next

        v5 = self.board[v + LEFTUP]
        v6 = self.board[v + LEFTDOWN]
        v7 = self.board[v + RIGHTUP]
        v8 = self.board[v + RIGHTDOWN]
        eye = (v1-e) * (v2-e) * (v3-e) * (v4-e) * (v5-e) * (v6-e) * (v7-e) * (v8-e) * v5 * v6 * v7 * v8

        if eye != 0:
            return False
        
        return True

    def capture(self, c, v, n):
        global FLAG
        FLAG += 1
        FRONTIER[n] = v
        FLAGS[v] = FLAG
        n1 = n
        n2 = n + 1

        while n2 > n1:
            i = FRONTIER[n1]
            n1 += 1

            m = i + UP
            s = self.board[m]
            if s == EMPTY:
                return n
            elif s == c and FLAGS[m] != FLAG:
                FLAGS[m] = FLAG
                FRONTIER[n2] = m
                n2 += 1

            m = i + DOWN
            s = self.board[m]
            if s == EMPTY:
                return n
            elif s == c and FLAGS[m] != FLAG:
                FLAGS[m] = FLAG
                FRONTIER[n2] = m
                n2 += 1

            m = i + LEFT
            s = self.board[m]
            if s == EMPTY:
                return n
            elif s == c and FLAGS[m] != FLAG:
                FLAGS[m] = FLAG
                FRONTIER[n2] = m
                n2 += 1

            m = i + RIGHT
            s = self.board[m]
            if s == EMPTY:
                return n
            elif s == c and FLAGS[m] != FLAG:
                FLAGS[m] = FLAG
                FRONTIER[n2] = m
                n2 += 1

        i = n
        codes = CODE_BLACK if c == BLACK else CODE_WHITE
        while i < n2:
            coord = FRONTIER[i]
            self.board[coord] = EMPTY
            self.hash_code ^= codes[coord]
            i += 1

        return n2

    def copy_board(self, board):
        for v in COORDS:
            self.board[v] = board[v]

    def copy(self, pos):
        self.next = pos.next
        self.ko = pos.ko
        self.hash_code = pos.hash_code
        self.vertex = pos.vertex
        self.pass_num = pos.pass_num
        self.copy_board(pos.board)

    def toJSON(self):
        JSON = {'board':self.board, 'next':self.next, 'ko':self.ko, 'vertex':self.vertex, 'pass_num':self.pass_num}
        return json.dumps(JSON)

    def fromJSON(self, JSON):
        self.copy_board(JSON['board'])
        self.next = JSON['next']
        self.ko = JSON['ko']
        self.vertex = JSON['vertex']
        self.pass_num = JSON['pass_num']
        self.init_hash_code()

    def move2(self, pos, v):
        global KO, NEXT

        if v == 0:
            pos.ko = 0
            pos.next = -self.next
            pos.vertex = v
            pos.hash_code = self.hash_code ^ CODE_KO[self.ko] ^ CODE_SWAP
            pos.pass_num = self.pass_num + 1
            if pos is not self:
                pos.copy_board(self.board)
            return True

        if v == self.ko or self.board[v] != EMPTY:
            return False

        if pos is not self:
            pos.copy_board(self.board)
        pos.board[v] = self.next
        if self.next == BLACK:
            pos.hash_code = self.hash_code ^ CODE_BLACK[v]
        else:
            pos.hash_code = self.hash_code ^ CODE_WHITE[v]
        enemy = -self.next

        # clear captures
        n = 0
        kov = 0
        mu = v + UP
        cu = pos.board[mu]
        if cu == enemy:
            n = pos.capture(enemy, mu, n)
            kov += 1
        elif cu == WALL:
            kov += 1

        md = v + DOWN
        cd = pos.board[md]
        if cd == enemy:
            n = pos.capture(enemy, md, n)
            kov += 1
        elif cd == WALL:
            kov += 1

        ml = v + LEFT
        cl = pos.board[ml]
        if cl == enemy:
            n = pos.capture(enemy, ml, n)
            kov += 1
        elif cl == WALL:
            kov += 1

        mr = v + RIGHT
        cr = pos.board[mr]
        if cr == enemy:
            n = pos.capture(enemy, mr, n)
            kov += 1
        elif cr == WALL:
            kov += 1

        pos.ko = FRONTIER[0]

        #suicide
        if pos.is_suicide(self.next, v):
            return False

        if n != 1 or kov < 4:
            pos.ko = 0

        pos.next = enemy
        pos.hash_code ^= CODE_KO[pos.ko]
        pos.hash_code ^= CODE_SWAP
        pos.vertex = v
        pos.pass_num = 0

        return True

    def is_suicide(self, c, v):
        global FLAG
        FLAG += 1
        FRONTIER[0] = v
        FLAGS[v] = FLAG
        n = 1

        while n > 0:
            n -= 1
            i = FRONTIER[n]

            m = i + UP
            s = self.board[m]
            if s == EMPTY:
                return False
            elif s == c and FLAGS[m] != FLAG:
                FLAGS[m] = FLAG
                FRONTIER[n] = m
                n += 1

            m = i + DOWN
            s = self.board[m]
            if s == EMPTY:
                return False
            elif s == c and FLAGS[m] != FLAG:
                FLAGS[m] = FLAG
                FRONTIER[n] = m
                n += 1

            m = i + LEFT
            s = self.board[m]
            if s == EMPTY:
                return False
            elif s == c and FLAGS[m] != FLAG:
                FLAGS[m] = FLAG
                FRONTIER[n] = m
                n += 1

            m = i + RIGHT
            s = self.board[m]
            if s == EMPTY:
                return False
            elif s == c and FLAGS[m] != FLAG:
                FLAGS[m] = FLAG
                FRONTIER[n] = m
                n += 1

        return True

    def play(self, j, i):
        return self.move2(self, i * M + j)

    def find_empties_group(self, v):
        FRONTIER[0] = v
        FLAGS[v] = FLAG
        n = 1
        empties = 0
        # flags of EMPTY, BLACK, WALL, WHITE
        colors = [False, False, False, False]

        while n > 0:
            n -= 1
            m = FRONTIER[n]
            empties += 1

            i = m + UP
            c = self.board[i]
            colors[c] = True
            if c == EMPTY and FLAGS[i] != FLAG:
                FLAGS[i] = FLAG
                FRONTIER[n] = i
                n += 1

            i = m + DOWN
            c = self.board[i]
            colors[c] = True
            if c == EMPTY and FLAGS[i] != FLAG:
                FLAGS[i] = FLAG
                FRONTIER[n] = i
                n += 1

            i = m + LEFT
            c = self.board[i]
            colors[c] = True
            if c == EMPTY and FLAGS[i] != FLAG:
                FLAGS[i] = FLAG
                FRONTIER[n] = i
                n += 1

            i = m + RIGHT
            c = self.board[i]
            colors[c] = True
            if c == EMPTY and FLAGS[i] != FLAG:
                FLAGS[i] = FLAG
                FRONTIER[n] = i
                n += 1

        if colors[BLACK]:
            if colors[WHITE]:
                return 0
            else:
                return empties
        else:
            if colors[WHITE]:
                return -empties
            else:
                return 0

    def score(self):
        global FLAG
        FLAG += 1
        score = -KOMI
        for i in COORDS:
            c = self.board[i]
            if c == EMPTY and FLAGS[i] != FLAG:
                score += self.find_empties_group(i)
            else:
                score += c

        return score

    def result(self):
        s = self.score()
        if s > 0:
            return 'B+' + '%.1f' % s
        elif s < 0:
            return 'W+' + '%.1f' % abs(s)
        else:
            return 'DRAW'

    def text(self):
        i = N
        s = "\n"
        while i > 0:
            s += str(i).zfill(2) + " "
            i -= 1
            j = 0
            while j < N:
                c = self.board[COORDS[i*N+j]]
                j += 1
                if c == BLACK:
                    s += "X "
                elif c == WHITE:
                    s += "O "
                else:
                    s += "+ "
            s += "\n"

        s += "   "
        while i < N:
            s += "{} ".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[i])
            i += 1
            
        s += "\n"
        
        return s

    def debug(self):
        print(self.text())


def init(n):
    global N, M, LN, LM, UP, DOWN, LEFTUP, LEFTDOWN, RIGHTUP, RIGHTDOWN, FLAGS, EMPTY_BOARD, COORDS, FRONTIER, FLAG
    global POSITION, POSITION_POOL, SIM_POS, MOVE_POS, BRANCH_SIM, HASH_SIM, CODE_WHITE, CODE_BLACK, CODE_KO
    global INPUT_BOARD, FLAG_BOARD, HASH_BOARD
    N = n
    M = N + 1
    LN = N * N
    LM = M * (M + 1) + 1
    UP = M
    DOWN = -M
    LEFTUP = LEFT + UP
    LEFTDOWN = LEFT + DOWN
    RIGHTUP = RIGHT + UP
    RIGHTDOWN = RIGHT + DOWN

    FLAGS = [0] * LM
    EMPTY_BOARD = [0] * LM
    FRONTIER = [0] * LN
    FLAG = 0
    KO = 0

    for i in range(M):
        EMPTY_BOARD[i] = WALL
        EMPTY_BOARD[LM - 1 - i] = WALL
        EMPTY_BOARD[(i + 1) * M] = WALL

    HASH_SIM = {}
    BRANCH_SIM = set()
    vlen = M * M
    CODE_WHITE = [0] * vlen
    CODE_BLACK = [0] * vlen
    CODE_KO = [0] * vlen

    COORDS = []
    for i in range(1, M):
        for j in range(1, M):
            v = i * M + j
            COORDS.append(v)
            CODE_WHITE[v] = random.getrandbits(64)
            CODE_BLACK[v] = random.getrandbits(64)
            CODE_KO[v] = random.getrandbits(64)

    INPUT_BOARD = torch.zeros(1, 2, N, N)
    FLAG_BOARD = [True] * LM
    HASH_BOARD = [False] * LM

    POSITION = Position()
    SIM_POS = Position()
    MOVE_POS = Position()
    POSITION_POOL = []
    i = 0
    while i < 100000:
        POSITION_POOL.append(Position())
        i += 1

init(7)

def clear():
    POSITION.copy_board(EMPTY_BOARD)
    POSITION.next = BLACK
    POSITION.ko = 0
    SIM_POS.copy(POSITION)
    MOVE_POS.copy(POSITION)
    HASH_SIM = {}
    BRANCH_SIM.clear()


def toXY(vertex):
    j = vertex % M
    i = int(vertex / M)
    return j, i

def get_positions(position):
    positions = []

    pos = POSITION_POOL.pop()
    for v in COORDS:
        if position.resonable(v) and position.move2(pos, v):
            positions.append(pos)
            pos = POSITION_POOL.pop()
    POSITION_POOL.append(pos)
    
    return positions

def get_captures(last_pos, next_pos):
    captures = []
    for v in COORDS:
        c1 = last_pos.board[v]
        c2 = next_pos.board[v]
        if c1 != EMPTY and c2 == EMPTY:
            captures.append(v)
    return captures


def text_flag_board():
    i = N
    s = "\n"
    while i > 0:
        s += str(i).zfill(2) + " "
        i -= 1
        j = 0
        while j < N:
            if FLAG_BOARD[COORDS[i*N+j]]:
                s += "O "
            else:
                s += ". "
            j += 1
        s += "\n"

    s += "   "
    while i < N:
        s += "{} ".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[i])
        i += 1
        
    s += "\n"
    
    return s
