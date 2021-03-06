#ifndef __POSITION_H__
#define __POSITION_H__

#include "Group.h"
#include "rapidjson/document.h"

class Position
{
  public:
    Position();
    ~Position();

    Position *move(int);

    float territory(int);

    float score();

    int passCount();

    void clear();

    void release();

    rapidjson::Value toJSON(rapidjson::Document::AllocatorType &);

    inline int vertex()
    {
        return _vertex;
    }

    inline int8_t next()
    {
        return _next;
    }

    inline int8_t boardi(int i)
    {
        return _board[i];
    }

    inline int ko()
    {
        return _ko;
    }

    inline void parent(Position *pos)
    {
        _parent = pos;
    }

    inline Position *parent()
    {
        return _parent;
    }

    void getChildren(std::vector<Position *> &);

    inline uint64_t hashCode()
    {
        return _hashCode;
    }

    void resetLiberty();

    inline int getSteps()
    {
        int step = 0;
        Position *p = _parent;
        while (p != nullptr)
        {
            step++;
            p = p->_parent;
        }
        return step;
    }

    void updateGroup();

    void debug();

    void debugGroup();

  private:
    int8_t _next;
    int _ko;
    int8_t *_board;
    Group **_group;
    int _vertex;
    uint64_t _hashCode;
    Position *_parent;
};

#endif