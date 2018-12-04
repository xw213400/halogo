#ifndef __DTLAYER_H__
#define __DTLAYER_H__

#include "Player.h"
#include "DTNode.h"
#include "Pool.h"
#include "tf/DTResnet.h"

class DTPlayer : public Player
{
  public:
    DTPlayer(const std::string&, int);

    virtual ~DTPlayer() {}

    virtual bool move();

    virtual void clear();

    static Pool<DTNode> POOL;

  private:
    float alphabeta(DTNode *, int, float, float);

    DTResnet *_net;

    DTNode *_best;

    int _depth;
};

#endif