#ifndef __MCTSNODE_H__
#define __MCTSNODE_H__

#include <vector>
#include "Position.h"
#include "Policy.h"

class MCTSNode
{
  public:
    MCTSNode();
    ~MCTSNode();

    void init(Policy *, MCTSNode *, Position *);

    MCTSNode *select();

    MCTSNode *expand();

    float simulate();

    void backpropagation(float);

    void release(bool recursive = true);

    inline Position *position()
    {
        return _position;
    }

    inline void position(Position *position)
    {
        _position = position;
    }

    inline std::vector<MCTSNode *> &children()
    {
        return _children;
    }

    inline std::vector<Position *> &positions()
    {
        return _positions;
    }

    inline MCTSNode *getParent()
    {
        return _parent;
    }

    inline float getScore()
    {
        return score;
    }

    inline float getQ()
    {
        return Q;
    }

    inline int getN()
    {
        return N;
    }

  private:
    Policy *_policy;
    MCTSNode *_parent;
    Position *_position;
    std::vector<Position *> _positions;
    int leaves;
    std::vector<MCTSNode *> _children;
    float Q, U, score;
    int N;
};

#endif