#ifndef __GO_H__
#define __GO_H__

#include "Pool.h"
#include "Position.h"

namespace go
{
extern const int8_t WHITE;
extern const int8_t EMPTY;
extern const int8_t BLACK;
extern const int8_t WALL;
extern const int PASS;

// # # # # #
// # + + +
// # + + +
// # + + +
// # # # #
extern const int N, LN, M, LM, LV, UP, DOWN, LEFT, RIGHT, LEFTUP, LEFTDOWN, RIGHTUP, RIGHTDOWN;
extern const float KOMI;
extern int FLAG;
extern int *FLAGS;
extern int8_t *EMPTY_BOARD;
extern int *COORDS;
extern int *FRONTIER;

extern uint64_t *CODE_WHITE;
extern uint64_t *CODE_BLACK;
extern uint64_t *CODE_KO;
extern uint64_t CODE_SWAP;

extern Pool<Position> POSITION_POOL;
extern Position *POSITION;
extern Group *GROUP_FLAG;

void init();

void clear();

std::pair<int, int> toJI(int);

bool isTrunk(Position *);

} // namespace go

#endif
