#ifndef __DTLAYER_H__
#define __DTLAYER_H__

#include "Player.h"
#include "DTNode.h"
#include "Pool.h"

class DTPlayer : public Player
{
  public:
    DTPlayer(Policy *);

    virtual ~DTPlayer() {}

    virtual bool move();

    virtual void clear();

    static Pool<DTNode> POOL;

  private:
    int _sims;

    Policy *_policy;

    DTNode *_best;
};

#endif