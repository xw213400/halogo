
#include "include/go.h"

using namespace std;
using namespace go;

Group::Group(void) : length(0), liberty(-1)
{
    stones = new int[LN];
}

Group::~Group(void)
{
    delete stones;
}

int Group::getLiberty(uint8_t *board)
{
    if (liberty != -1)
    {
        return liberty;
    }

    FLAG += 1;
    liberty = 0;

    for (int i = 0; i != length; ++i)
    {
        int s = stones[i];

        int v = s + UP;
        uint8_t c = board[v];
        if (c == EMPTY && FLAGS[v] != FLAG)
        {
            FLAGS[v] = FLAG;
            ++liberty;
            if (liberty >= 2)
            {
                return liberty;
            }
        }

        v = s + DOWN;
        c = board[v];
        if (c == EMPTY && FLAGS[v] != FLAG)
        {
            FLAGS[v] = FLAG;
            ++liberty;
            if (liberty >= 2)
            {
                return liberty;
            }
        }

        v = s + LEFT;
        c = board[v];
        if (c == EMPTY && FLAGS[v] != FLAG)
        {
            FLAGS[v] = FLAG;
            ++liberty;
            if (liberty >= 2)
            {
                return liberty;
            }
        }

        v = s + RIGHT;
        c = board[v];
        if (c == EMPTY && FLAGS[v] != FLAG)
        {
            FLAGS[v] = FLAG;
            ++liberty;
            if (liberty >= 2)
            {
                return liberty;
            }
        }
    }
}

Position::Position(void)
{
    board = new uint8_t[LM];
    memcpy(board, EMPTY_BOARD, LM);
    group = static_cast<Group**>(malloc(sizeof(nullptr) * LM));
    memset(group, nullptr, LM);
    mygroup = new Group();
    vertex = PASS;
    hash_code = 0;
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
    if (v == PASS)
    {
        Position* pos = POSITION_POOL.pop();
    }
}