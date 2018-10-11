
#include "go.h"

using namespace std;
using namespace go;

void init(int n, float komi)
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
    EMPTY_BOARD = new uint8_t[LM];

    CODE_SWAP = mt_rand();

    if (POSITION == nullptr)
    {
        POSITION = POSITION_POOL.pop();
    }
    else
    {
        POSITION->clear();
    }
}