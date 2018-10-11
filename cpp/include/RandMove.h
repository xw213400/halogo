#ifndef __RANDMOVE_H__
#define __RANDMOVE_H__

#include <map>
#include "Policy.h"

class RandMove : public Policy
{
  public:
    RandMove(float);

    ~RandMove();

    Position **get(Position *);

    float sim(Position *);

    void clear();

  private:
    std::map<unsigned long long, float> hash;
};

#endif