#ifndef __GO_H__
#define __GO_H__

#include <random>
#include "pool.h"

using namespace std;

namespace go
{
const uint8_t WHITE = -1;
const uint8_t EMPTY = 0;
const uint8_t BLACK = 1;
const uint8_t WALL = 2;
const int PASS = 0;

// # # # # #
// # + + +
// # + + +
// # + + +
// # # # #
int N, LN, M, LM, UP, DOWN, LEFT, RIGHT, LEFTUP, LEFTDOWN, RIGHTUP, RIGHTDOWN;
float KOMI;
int FLAG = 0;
int *FLAGS = new int[LM];
uint8_t *EMPTY_BOARD;
int *COORDS;
int *FRONTIER;
int *POSITION;

mt19937_64 mt_rand(time(0));

unsigned long long *CODE_WHITE = nullptr;
unsigned long long *CODE_BLACK = nullptr;
unsigned long long *CODE_KO = nullptr;
unsigned long long CODE_SWAP = mt_rand();

void init(int n, float komi = 5.5)
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
}

class Group
{
  public:
    Group();
    ~Group();

    int getLiberty(uint8_t *);

    int *stones;
    int length;
    int liberty;
};

class Position
{
  public:
    Position();
    ~Position();

    Position *move(int);

  private:
    int next = BLACK;
    int ko = 0;

    uint8_t *board = nullptr;
    Group **group = nullptr;
    Group *mygroup = nullptr;

    int vertex = 0;
    unsigned long long hash_code = 0;
    Position *parent = nullptr;
};

Pool<Position> POSITION_POOL;
// INPUT_BOARD = None;
Position *TRUNK = nullptr;

} // namespace go

#endif
