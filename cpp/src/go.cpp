
#include <random>
#include "go.h"
#include "MCTSPlayer.h"

using namespace std;
using namespace tensorflow;

namespace go
{

const int8_t WHITE = -1;
const int8_t EMPTY = 0;
const int8_t BLACK = 1;
const int8_t WALL = 2;
const int PASS = 0;

const int N = 9;
const float KOMI = 5.5f;
const int LN = N * N;
const int M = N + 1;
const int LM = M * (M + 1) + 1;
const int LV = M * M;

const int LEFT = -1;
const int RIGHT = 1;
const int UP = M;
const int DOWN = -M;

const int LEFTUP = LEFT + UP;
const int LEFTDOWN = LEFT + DOWN;
const int RIGHTUP = RIGHT + UP;
const int RIGHTDOWN = RIGHT + DOWN;

int FLAG = 0;
int *FLAGS = new int[LM];
int8_t *EMPTY_BOARD = new int8_t[LM];
int *COORDS = new int[LN];
int *FRONTIER = new int[LN];

uint64_t *CODE_WHITE = new uint64_t[LV];
uint64_t *CODE_BLACK = new uint64_t[LV];
uint64_t *CODE_KO = new uint64_t[LV];
uint64_t CODE_SWAP = 0;

Pool<Position> POSITION_POOL("POSITION");
Position *POSITION = nullptr;
Group *GROUP_FLAG = new Group();

Tensor INPUT_BOARD(DT_FLOAT, {1, go::N, go::N, 1});

} // namespace go

void go::init()
{
    memset(FLAGS, 0, sizeof(int) * LM);
    memset(EMPTY_BOARD, 0, sizeof(int8_t) * LM);

    for (int i = 0; i != M; ++i)
    {
        EMPTY_BOARD[i] = WALL;
        EMPTY_BOARD[LM - 1 - i] = WALL;
        EMPTY_BOARD[(i + 1) * M] = WALL;
    }

    srand((unsigned)time(NULL));
    mt19937_64 mt_rand(time(0));

    CODE_SWAP = mt_rand();

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

    POSITION_POOL.resize(1000000);
    MCTSPlayer::POOL.resize(50000);
    Group::POOL.resize(50000);
    POSITION = POSITION_POOL.pop();
    POSITION->clear();
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
        trunk = trunk->parent();
    }
    return false;
}

void go::clear(void)
{
    Position *p = POSITION->parent();
    while (p != nullptr)
    {
        p->release();
        p = p->parent();
    }

    POSITION->clear();
}