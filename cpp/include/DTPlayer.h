#ifndef __DTLAYER_H__
#define __DTLAYER_H__

#include "Player.h"
#include "DTNode.h"
#include "Moves.h"
#include "tf/DTResnet.h"

class DTPlayer : public Player
{
  public:
    DTPlayer(const std::string&, int, int);

    virtual ~DTPlayer() {}

    virtual bool move();

    virtual void clear();

  private:
    float alphabeta(DTNode *, int, float, float);

    DTResnet *_net;

    int _depth;

    int _branches;

    std::vector<Moves<DTNode>> _nodes;
};

#endif