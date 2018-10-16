#ifndef __RANDMOVE_H__
#define __RANDMOVE_H__

#include <unordered_map>
#include "Policy.h"

class RandMove : public Policy
{
  public:
    RandMove(float);

    virtual ~RandMove();

    void get(Position *, std::vector<Position*>&);

    float sim(Position *);

    void clear();

  private:
    std::unordered_map<uint64_t, float> hash;
};

#endif