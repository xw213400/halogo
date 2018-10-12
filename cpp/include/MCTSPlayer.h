#ifndef __MCTSPLAYER_H__
#define __MCTSPLAYER_H__

#include "Policy.h"
#include "MCTSNode.h"
#include "Pool.h"

class MCTSPlayer
{
public:
  MCTSPlayer(float, Policy *);

  ~MCTSPlayer();

  bool move();

  void clear();

  static Pool<MCTSNode> pool;

private:
  float seconds;

  Policy *policy;

  MCTSNode *best;
};

#endif