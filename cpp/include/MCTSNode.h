#ifndef __MCTSNODE_H__
#define __MCTSNODE_H__

#include <vector>
#include "Position.h"

class MCTSNode
{
  public:
    MCTSNode();
    ~MCTSNode();

    void init(MCTSNode *, Position *);

    MCTSNode *select();

    void expand();

    float simulate();

    void backpropagation(float);

    void release(bool);

  private:
    void add_leaf(int);

    void sub_leaf();

    MCTSNode *parent;
    Position *position;
    std::vector<Position*> positions;
    int leaves;
    std::vector<MCTSNode*> children;
    float Q, U, score;
    int N;
};

#endif