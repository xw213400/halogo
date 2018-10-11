#include <cstring>
#include "Position.h"
#include "go.h"

Position::Position(void)
{
    board = new unsigned char[go::LM];
    memcpy(board, go::EMPTY_BOARD, go::LM);
    group = static_cast<Group**>(malloc(sizeof(nullptr) * go::LM));
    memset(group, 0, sizeof(nullptr) * go::LM);
    mygroup = new Group();
    vertex = go::PASS;
    hash_code = 0;
    ko = 0;
    next = go::BLACK;
    parent = nullptr;
}

Position::~Position(void)
{
    delete board;
    delete group;
    delete mygroup;
}

Position* Position::move(int v)
{
    if (v == go::PASS)
    {
        Position* pos = go::POSITION_POOL.pop();
    }
}