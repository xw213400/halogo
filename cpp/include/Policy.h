#ifndef __POLICY_H__
#define __POLICY_H__

#include "Position.h"

class Policy
{
  public:
    Policy(float);

    virtual ~Policy() = 0;

    virtual Position **get(Position *) = 0;

    virtual float sim(Position *) = 0;

    virtual void clear();

    inline float getPUCT() {
        return PUCT;
    }

private:
    float PUCT;
};

#endif