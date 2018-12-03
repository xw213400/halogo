#ifndef __MCTSPLAYER_H__
#define __MCTSPLAYER_H__

#include "Player.h"
#include "MCTSNode.h"
#include "Pool.h"

class MCTSPlayer : public Player
{
  public:
    MCTSPlayer(Policy *, int);

    virtual ~MCTSPlayer() {}

    virtual bool move();

    virtual void clear();

    static Pool<MCTSNode> POOL;

  private:
    int _sims;

    MCTSNode *_best;
};

#endif