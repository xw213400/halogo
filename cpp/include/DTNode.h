#ifndef __DTNODE_H__
#define __DTNODE_H__

#include <vector>
#include "Position.h"

class DTNode
{
  public:
    DTNode();

    ~DTNode();

    void init(Position *, float);

    static void Release();

    inline float evaluate()
    {
        return _evaluate;
    }

    inline Position *position()
    {
        return _position;
    }

  private:
    Position *_position;
    float _evaluate;

    static std::vector<Position*> POSITIONS;
};

#endif