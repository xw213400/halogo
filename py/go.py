
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
N = 9
LN = 81
M = 10
LM = 10 * 11 + 1

UP = M
DOWN = -M

LEFTUP = LEFT + UP
LEFTDOWN = LEFT + DOWN
RIGHTUP = RIGHT + UP
RIGHTDOWN = RIGHT + DOWN

NEIGHBORS = [UP, DOWN, LEFT, RIGHT]

KOMI = 5.5
FLAG = 0
FLAGS = None
EMPTY_BOARD = None
COORDS = None
FRONTIER = None
POSITION = None
POSITION_POOL = None

CODE_WHITE = None
CODE_BLACK = None
CODE_KO = None
CODE_SWAP = random.getrandbits(64)

INPUT_BOARD = None
TRUNK = None

class Group:
    def __init__(self, v):
        self.stones = [v]
        self.liberty = -1

    def get_liberty(self, board):
        global FLAG

        if self.liberty != -1:
            return self.liberty

        FLAG += 1
        self.liberty = 0
        for s in self.stones:
            for n in NEIGHBORS:
                v = s + n
                c = board[v]
                if c == EMPTY and FLAGS[v] != FLAG:
                    FLAGS[v] = FLAG
                    self.liberty = self.liberty + 1
                    if self.liberty >= 2:
                        return self.liberty

        return self.liberty



class Position:
    def __init__(self):
        self.next = BLACK
        self.ko = 0
        self.board = EMPTY_BOARD[:]
        self.group = [None] * LM
        self.vertex = 0
        self.hash_code = 0
        self.parent = None

    # prepare input plane for resnet
    # INPUT_BOARD[0]: enemy:-1, empty:0, self:1
    # INPUT_BOARD[1]: unresonable:0, resonable:1, ko:-1
    def input_board(self):
        global POSITION_POOL, INPUT_BOARD

        c = 0
        x = 0
        y = 0
        while c < LN:
            v = COORDS[c]
            p = 0

            pos = self.move(v)
            if pos is not None:
                p = 1
                POSITION_POOL.append(pos)

            if v == self.ko:
                p = -1

            INPUT_BOARD[0, 0, y, x] = self.board[v] * self.next
            INPUT_BOARD[0, 1, y, x] = p

            x += 1
            if x == N:
                y += 1
                x = 0
            c = y * N + x

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

    def toJSON(self):
        BOARD = []
        for v in COORDS:
            BOARD.append(board[v])
        JSON = {'board':BOARD, 'next':self.next, 'ko':self.ko, 'vertex':self.vertex}
        return json.dumps(JSON)

    def fromJSON(self, JSON):
        board = JSON['board']
        i = 0
        while i < LN:
            self.board[COORDS[i]] = board[i]
            i += 1
        self.next = JSON['next']
        self.ko = JSON['ko']
        self.vertex = JSON['vertex']
        self.init_hash_code()

    def move(self, v):
        global KO, NEXT, POSITION_POOL, FRONTIER

        if v == 0:
            pos = POSITION_POOL.pop()

            pos.parent = self
            pos.ko = 0
            pos.next = -self.next
            pos.vertex = v
            pos.hash_code = self.hash_code ^ CODE_KO[self.ko] ^ CODE_SWAP
            for s in COORDS:
                pos.board[s] = self.board[s]

            return pos

        if self.board[v] != EMPTY or v == self.ko:
            return None

        resonable = False
        ko = 0

        vu = v + UP
        vd = v + DOWN
        vl = v + LEFT
        vr = v + RIGHT

        cu = self.board[vu]
        cd = self.board[vd]
        cl = self.board[vl]
        cr = self.board[vr]

        gs = []

        gu = self.group[vu]
        if gu is not None:
            gs.append(gu)
            
        gd = self.group[vd]
        if gd is not None and gd != gu:
            gs.append(gd)

        gl = self.group[vl]
        if gl is not None and gl != gu and gl != gd:
            gs.append(gl)

        gr = self.group[vr]
        if gr is not None and gr != gu and gr != gd and gr != gl:
            gs.append(gr)

        ee = -self.next

        if cu * cd * cl * cr == 0: #neighbor empty
            resonable = True

        nTakes = 0

        for g in gs:
            color = self.board[g.stones[0]]
            if color == ee:
                liberty = g.get_liberty(self.board)
                if liberty == 1:
                    for s in g.stones:
                        FRONTIER[nTakes] = s
                        nTakes += 1
            else:
                if not resonable: # around stone and wall
                    liberty = g.get_liberty(self.board)
                    if liberty > 1: # not suicide
                        resonable = True

        if nTakes > 0:
            if not resonable and nTakes == 1 and (cu+ee)*(cd+ee)*(cl+ee)*(cr+ee) != 0: # KO
                ko = FRONTIER[0]
            resonable = True

        if (cu-ee)*(cd-ee)*(cl-ee)*(cr-ee) != 0 and len(gs) == 1:#eye
            resonable = False

        if not resonable:
            return None

        hash_code = self.hash_code

        if self.next == BLACK:
            hash_code ^= CODE_BLACK[v] ^ CODE_KO[self.ko] ^ CODE_SWAP ^ CODE_KO[ko]
        else:
            hash_code ^= CODE_WHITE[v] ^ CODE_KO[self.ko] ^ CODE_SWAP ^ CODE_KO[ko]

        codes = CODE_BLACK if ee == BLACK else CODE_WHITE
        i = 0
        while i < nTakes:
            hash_code ^= codes[FRONTIER[i]]
            i += 1

        # check loop position
        p = self
        while p is not None:
            if p.hash_code == hash_code:
                return None
            else:
                p = p.parent

        pos = POSITION_POOL.pop()

        for s in COORDS:
            pos.board[s] = self.board[s]
        pos.board[v] = self.next
        pos.next = -self.next
        pos.vertex = v
        pos.hash_code = hash_code
        pos.ko = ko

        i = 0
        while i < nTakes:
            pos.board[FRONTIER[i]] = EMPTY
            i += 1
        pos.parent = self

        return pos

    def pass_count(self):
        pc = 0
        if self.vertex == 0:
            pc += 1
        if self.parent is not None and self.parent.vertex == 0:
            pc += 1
        return pc

    def territory(self, v):
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

    def update_group(self):
        parent = self.parent

        if parent is not None:
            for v in COORDS:
                g = self.group[v] = parent.group[v]
                if g is not None:
                    self.group[v].liberty = -1

            if self.vertex != 0:
                g = self.group[self.vertex] = Group(self.vertex)
                ns = [self.vertex+UP, self.vertex+DOWN, self.vertex+LEFT, self.vertex+RIGHT]
                for n in ns:
                    gg = self.group[n]
                    if self.board[n] == parent.next and gg != g:
                        for s in gg.stones:
                            self.group[s] = g
                        g.stones.extend(gg.stones)
                    elif gg is not None and self.board[n] == EMPTY:
                        for s in gg.stones:
                            self.group[s] = None

    def get_children(self):
        positions = []

        self.update_group()

        for v in COORDS:
            pos = self.move(v)
            if pos is not None:
                positions.append(pos)
        
        return positions

    def score(self):
        global FLAG
        FLAG += 1
        score = 0 #-KOMI
        for i in COORDS:
            c = self.board[i]
            if c == EMPTY and FLAGS[i] != FLAG:
                score += self.territory(i)
            else:
                score += c

        return score

    def result(self):
        s = self.score() - KOMI
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

    def debug_group(self):
        i = N
        s = "\n"
        while i > 0:
            s += str(i).zfill(2) + " "
            i -= 1
            j = 0
            while j < N:
                v = COORDS[i*N+j]
                c = self.board[v]
                g = self.group[v]
                j += 1
                if c == BLACK and g is not None:
                    s += "X "
                elif c == WHITE and g is not None:
                    s += "O "
                else:
                    s += "+ "
            s += "\n"

        s += "   "
        while i < N:
            s += "{} ".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[i])
            i += 1
            
        s += "\n"
        print(s)


def init(n):
    global N, M, LN, LM, UP, DOWN, LEFTUP, LEFTDOWN, RIGHTUP, RIGHTDOWN, FLAGS, EMPTY_BOARD, COORDS, FRONTIER, FLAG, NEIGHBORS
    global POSITION, POSITION_POOL, CODE_WHITE, CODE_BLACK, CODE_KO, TRUNK
    global INPUT_BOARD
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
    NEIGHBORS = [UP, DOWN, LEFT, RIGHT]

    FLAGS = [0] * LM
    EMPTY_BOARD = [0] * LM
    FRONTIER = [0] * LN
    FLAG = 0
    KO = 0

    for i in range(M):
        EMPTY_BOARD[i] = WALL
        EMPTY_BOARD[LM - 1 - i] = WALL
        EMPTY_BOARD[(i + 1) * M] = WALL

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

    POSITION_POOL = []
    i = 0
    while i < 1000000:
        POSITION_POOL.append(Position())
        i += 1

    POSITION = POSITION_POOL.pop()
    TRUNK = set([POSITION])

def clear():
    global POSITION, TRUNK, POSITION_POOL
    for s in COORDS:
        POSITION.board[s] = EMPTY_BOARD[s]
        POSITION.group[s] = None
    POSITION.next = BLACK
    POSITION.ko = 0
    POSITION.vertex = 0
    POSITION.hash_code = 0

    p = POSITION.parent
    while p is not None:
        POSITION_POOL.append(p)
        p = p.parent

    POSITION.parent = None
    TRUNK = set([POSITION])

def toJI(v):
    j = v % M
    i = int(v / M)
    return j, i

def toV(i, j):
    return i * M + j

def toXY(c):
    x = c % N
    y = int(c / N)
    return x, y

def toC(x, y):
    return y * N + x

def get_take(position):
    take = []
    for v in COORDS:
        c1 = position.parent.board[v]
        c2 = position.board[v]
        if c1 != EMPTY and c2 == EMPTY:
            take.append(v)
    return take

def print_input(self):
    i = N
    s = "\n"
    while i > 0:
        s += str(i).zfill(2) + " "
        i -= 1
        j = 0
        while j < N:
            s += str(int(INPUT_BOARD[0, 0, i, j])) + " "
            j += 1
        s += "\n"

    s += "   "
    while i < N:
        s += "{} ".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[i])
        i += 1

    i = N
    s += "\n"
    while i > 0:
        s += str(i).zfill(2) + " "
        i -= 1
        j = 0
        while j < N:
            s += str(int(INPUT_BOARD[0, 1, i, j])) + " "
            j += 1
        s += "\n"

    s += "   "
    while i < N:
        s += "{} ".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[i])
        i += 1

    s += "\n"
    print(s)

def move(v):
    global POSITION, TRUNK

    pos = POSITION.move(v)
    if pos is not None:
        POSITION = pos
        POSITION.update_group()
        TRUNK = set()
        while pos is not None:
            TRUNK.add(pos)
            pos = pos.parent

        return True
    
    return False

def get_step():
    step = 0
    p = POSITION.parent
    while p is not None:
        step += 1
        p = p.parent
    
    return step
    