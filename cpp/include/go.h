#ifndef __GO_H__
#define __GO_H__

#include <random>
#include "Pool.h"
#include "Position.h"

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

std::mt19937_64 mt_rand(time(0));

unsigned long long *CODE_WHITE = nullptr;
unsigned long long *CODE_BLACK = nullptr;
unsigned long long *CODE_KO = nullptr;
unsigned long long CODE_SWAP;

Pool<Position> POSITION_POOL(1000000);
Position *POSITION = nullptr;

void init(int, float komi = 5.5f);

void clear();

} // namespace go

#endif
