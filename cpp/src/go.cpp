
#include <random>
#include "go.h"

using namespace std;

namespace go
{

const int8_t WHITE = -1;
const int8_t EMPTY = 0;
const int8_t BLACK = 1;
const int8_t WALL = 2;
const int PASS = 0;

int N, LN, M, LM, UP, DOWN, LEFT, RIGHT, LEFTUP, LEFTDOWN, RIGHTUP, RIGHTDOWN;
float KOMI = 5.5f;
int FLAG = 0;
int *FLAGS = nullptr;
int8_t *EMPTY_BOARD = nullptr;
int *COORDS = nullptr;
int *FRONTIER = nullptr;

uint64_t *CODE_WHITE = nullptr;
uint64_t *CODE_BLACK = nullptr;
uint64_t *CODE_KO = nullptr;
uint64_t CODE_SWAP = 0;

Pool<Position> POSITION_POOL(1000000);
Position *POSITION = nullptr;

} // namespace go

void go::init(int n, float komi)
{
    N = n;
    KOMI = komi;

    LN = N * N;
    M = N + 1;
    LM = M * (M + 1) + 1;

    LEFT = -1;
    RIGHT = 1;
    UP = M;
    DOWN = -M;

    LEFTUP = LEFT + UP;
    LEFTDOWN = LEFT + DOWN;
    RIGHTUP = RIGHT + UP;
    RIGHTDOWN = RIGHT + DOWN;

    FLAG = 0;
    FLAGS = new int[LM];
    memset(FLAGS, 0, sizeof(int) * LM);
    EMPTY_BOARD = new int8_t[LM];
    memset(EMPTY_BOARD, 0, sizeof(int8_t) * LM);

    for (int i = 0; i != M; ++i)
    {
        EMPTY_BOARD[i] = WALL;
        EMPTY_BOARD[LM - 1 - i] = WALL;
        EMPTY_BOARD[(i + 1) * M] = WALL;
    }

    int vlen = M * M;

    CODE_WHITE = new uint64_t[vlen];
    CODE_BLACK = new uint64_t[vlen];
    CODE_KO = new uint64_t[vlen];

    mt19937_64 mt_rand(time(0));

    CODE_SWAP = mt_rand();

    COORDS = new int[LN];

    for (int i = 0; i != N; ++i)
    {
        for (int j = 0; j != N; ++j)
        {
            int s = i * N + j;
            int v = (i + 1) * M + j + 1;
            COORDS[s] = v;
            CODE_WHITE[v] = mt_rand();
            CODE_BLACK[v] = mt_rand();
            CODE_KO[v] = mt_rand();
        }
    }

    if (POSITION == nullptr)
    {
        POSITION = POSITION_POOL.pop();
    }
    else
    {
        POSITION->clear();
    }
}

pair<int, int> go::toJI(int v)
{
    int j = v % M;
    int i = v / M;
    return make_pair(j, i);
}

bool go::isTrunk(Position *position)
{
    Position *trunk = go::POSITION;
    while (trunk != nullptr)
    {
        if (trunk == position)
        {
            return true;
        }
        trunk = trunk->getParent();
    }
    return false;
}

void go::clear(void)
{
    Position *p = POSITION->getParent();
    while (p != nullptr)
    {
        POSITION_POOL.push(p);
        p = p->getParent();
    }

    POSITION->clear();
}