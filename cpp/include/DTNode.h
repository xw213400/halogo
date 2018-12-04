#ifndef __DTNODE_H__
#define __DTNODE_H__

#include <vector>
#include "Position.h"
#include "tf/DTResnet.h"

class DTResnet;

class DTNode
{
  public:
    DTNode();

    ~DTNode();

    void init(Position *, float);

    const std::vector<DTNode *> &expand(DTResnet *net);

    void release(bool recursive = true);

    inline float evaluate()
    {
        return _evaluate;
    }

    inline Position *position()
    {
        return _position;
    }

    inline void position(Position *position)
    {
        _position = position;
    }

    inline std::vector<DTNode *> &children()
    {
        return _children;
    }

    static const int BRANCHES = 20;

  private:
    Position *_position;
    std::vector<DTNode *> _children;
    float _evaluate;
};

#endif