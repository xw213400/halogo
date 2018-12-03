#ifndef __PLAYER_H__
#define __PLAYER_H__

#include "Policy.h"

class Player
{
  public:
    Player(Policy *policy) : _policy(policy) {}

    virtual ~Player() {}

    virtual bool move() = 0;

    virtual void clear() = 0;

  protected:
    Policy *_policy;
};

#endif