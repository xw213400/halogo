#ifndef __MCTSPLAYER_H__
#define __MCTSPLAYER_H__

#include "Policy.h"

class MCTSPlayer
{
  public:
    MCTSPlayer(float, Policy*);

    ~MCTSPlayer();

    bool move();

    void clear();

  private:
    float seconds_per_move;

    Policy* policy;
};

#endif