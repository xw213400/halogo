#ifndef __MCTSPLAYER_H__
#define __MCTSPLAYER_H__

#include "Policy.h"
#include "MCTSNode.h"
#include "Pool.h"

class MCTSPlayer
{
  public:
    MCTSPlayer(int, Policy *);

    ~MCTSPlayer();

    bool move();

    void clear();

    static Pool<MCTSNode> POOL;

  private:
    int _sims;

    Policy *_policy;

    MCTSNode *_best;
};

#endif