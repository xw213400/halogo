#ifndef __RANDMOVE_H__
#define __RANDMOVE_H__

#include <unordered_map>
#include "Policy.h"

class RandMove : public Policy
{
  public:
    RandMove(float);

    virtual ~RandMove(){};

    virtual void get(Position *, std::vector<Position *> &);

    virtual float sim(Position *);

    virtual void clear();

  private:
    std::unordered_map<uint64_t, float> _hash;
};

#endif