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

  inline Position *getPosition()
  {
    return position;
  }

  inline void setPosition(Position *pos)
  {
    position = pos;
  }

  inline std::vector<MCTSNode *> &getChildren()
  {
    return children;
  }

  inline std::vector<Position *> &getPositions()
  {
    return positions;
  }

  inline MCTSNode *getParent()
  {
    return parent;
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
  inline void addLeaf(int n)
  {
    leaves += n;
    if (parent != nullptr)
    {
      parent->addLeaf(n);
    }
  }

  inline void subLeaf()
  {
    --leaves;
    if (parent != nullptr)
    {
      parent->subLeaf();
    }
  }

  Policy *policy;
  MCTSNode *parent;
  Position *position;
  std::vector<Position *> positions;
  int leaves;
  std::vector<MCTSNode *> children;
  float Q, U, score;
  int N;
};

#endif