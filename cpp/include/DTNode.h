#ifndef __DTNODE_H__
#define __DTNODE_H__

#include <vector>
#include "Position.h"
#include "Policy.h"

class DTNode
{
  public:
    DTNode();
    ~DTNode();

    void init(Policy *, DTNode *, Position *);

    DTNode *select();

    DTNode *expand();

    void release(bool recursive = true);

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

    inline DTNode *getParent()
    {
        return _parent;
    }

  private:
    Policy *_policy;
    DTNode *_parent;
    Position *_position;
    std::vector<DTNode *> _children;
};

#endif