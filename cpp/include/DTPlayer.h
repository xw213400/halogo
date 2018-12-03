#ifndef __DTLAYER_H__
#define __DTLAYER_H__

#include "Policy.h"
#include "MCTSNode.h"
#include "Pool.h"

class DTPlayer
{
  public:
    DTPlayer(int, Policy *);

    ~DTPlayer();

    bool move();

    void clear();

    static Pool<MCTSNode> POOL;

  private:
    int _sims;

    Policy *_policy;

    MCTSNode *_best;
};

#endif