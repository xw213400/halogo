#include "Group.h"
#include "go.h"

Pool<Group> Group::POOL("GROUP");

Group* Group::get(int v) {
    Group* g = POOL.pop();

    g->_stones[g->_n++] = v;
    g->_rc = 1;

    return g;
};

Group::Group(void) :_liberty(-1), _rc(0)
{
    _stones = new int[go::LN];
}

Group::~Group(void)
{
}

void Group::resetLiberty(int8_t *board)
{
    go::FLAG++;
    _liberty = 0;

    for (size_t i = 0; i != _n; ++i)
    {
        int s = _stones[i];

        int v = s + go::UP;
        int8_t c = board[v];

        if (c == go::EMPTY && go::FLAGS[v] != go::FLAG)
        {
            go::FLAGS[v] = go::FLAG;
            ++_liberty;
            if (_liberty >= 2)
            {
                break;
            }
        }

        v = s + go::DOWN;
        c = board[v];
        if (c == go::EMPTY && go::FLAGS[v] != go::FLAG)
        {
            go::FLAGS[v] = go::FLAG;
            ++_liberty;
            if (_liberty >= 2)
            {
                break;
            }
        }

        v = s + go::LEFT;
        c = board[v];
        if (c == go::EMPTY && go::FLAGS[v] != go::FLAG)
        {
            go::FLAGS[v] = go::FLAG;
            ++_liberty;
            if (_liberty >= 2)
            {
                break;
            }
        }

        v = s + go::RIGHT;
        c = board[v];
        if (c == go::EMPTY && go::FLAGS[v] != go::FLAG)
        {
            go::FLAGS[v] = go::FLAG;
            ++_liberty;
            if (_liberty >= 2)
            {
                break;
            }
        }
    }
}