#include "Group.h"
#include "go.h"

Pool<Group> Group::pool;

Group* Group::get(int v) {
    Group* g = pool.pop();
    g->stones.push_back(v);
    g->rc = 1;
    return g;
};

Group::Group(void) :liberty(-1), rc(0)
{
    stones.reserve(go::LN);
}

Group::~Group(void)
{
}

int Group::getLiberty(int8_t *board)
{
    if (liberty != -1)
    {
        return liberty;
    }

    go::FLAG++;
    liberty = 0;

    size_t length = stones.size();
    for (size_t i = 0; i != length; ++i)
    {
        int s = stones[i];

        int v = s + go::UP;
        int8_t c = board[v];

        if (c == go::EMPTY && go::FLAGS[v] != go::FLAG)
        {
            go::FLAGS[v] = go::FLAG;
            ++liberty;
            if (liberty >= 2)
            {
                return liberty;
            }
        }

        v = s + go::DOWN;
        c = board[v];
        if (c == go::EMPTY && go::FLAGS[v] != go::FLAG)
        {
            go::FLAGS[v] = go::FLAG;
            ++liberty;
            if (liberty >= 2)
            {
                return liberty;
            }
        }

        v = s + go::LEFT;
        c = board[v];
        if (c == go::EMPTY && go::FLAGS[v] != go::FLAG)
        {
            go::FLAGS[v] = go::FLAG;
            ++liberty;
            if (liberty >= 2)
            {
                return liberty;
            }
        }

        v = s + go::RIGHT;
        c = board[v];
        if (c == go::EMPTY && go::FLAGS[v] != go::FLAG)
        {
            go::FLAGS[v] = go::FLAG;
            ++liberty;
            if (liberty >= 2)
            {
                return liberty;
            }
        }
    }

    return liberty;
}