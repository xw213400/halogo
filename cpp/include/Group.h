#ifndef __GROUP_H__
#define __GROUP_H__

#include <vector>
#include <stdint.h>
#include <cstring>
#include "Pool.h"

class Group
{
  public:
    Group();
    ~Group();

    inline int liberty()
    {
        return _liberty;
    }

    inline void liberty(int liberty)
    {
        _liberty = liberty;
    }

    inline void reference(int n)
    {
        _rc += n;
    }

    inline void release()
    {
        if (0 == _rc)
        {
            _liberty = -1;
            _n = 0;
            POOL.push(this);
        }
    }

    inline void merge(Group *g)
    {
        memcpy(_stones + _n, g->_stones, sizeof(int) * g->_n);
        _n += g->_n;
    }

    inline int8_t color(int8_t *board)
    {
        return board[_stones[0]];
    }

    inline int n()
    {
        return _n;
    }

    inline int getStone(int i)
    {
        return _stones[i];
    }

    void resetLiberty(int8_t *);

    static Group *get(int);

    static Pool<Group> POOL;

  private:
    int *_stones;
    int _liberty;
    int _rc;
    size_t _n;
};

#endif