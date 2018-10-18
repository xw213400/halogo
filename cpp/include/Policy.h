#ifndef __POLICY_H__
#define __POLICY_H__

#include "Position.h"

class Policy
{
  public:
    Policy(float PUCT) : _PUCT(PUCT){};

    virtual ~Policy() {}

    virtual void get(Position *, std::vector<Position *> &) = 0;

    virtual float sim(Position *) = 0;

    virtual void clear() = 0;

    inline float PUCT()
    {
        return _PUCT;
    }

  private:
    float _PUCT;
};

#endif