#ifndef __RANDMOVE_H__
#define __RANDMOVE_H__

#include <unordered_map>
#include "Policy.h"

class RandMove : public Policy
{
  public:
    RandMove(float, bool useScore = true);

    virtual ~RandMove(){};

    void get(Position *, std::vector<Position *> &);

    float sim(Position *);

    void clear();

    inline bool useScore() {
      return _useScore;
    }

  private:
    std::unordered_map<uint64_t, float> _hash;
    bool _useScore;
};

#endif