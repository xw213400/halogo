#ifndef __GROUP_H__
#define __GROUP_H__

#include <vector>
#include <stdint.h>
#include "Pool.h"

class Group
{
public:
  Group();
  ~Group();

  int getLiberty(int8_t *);

  std::vector<int> stones;
  int liberty;
  int rc;

  inline void release()
  {
    if (0 == rc)
    {
      liberty = -1;
      stones.clear();
      pool.push(this);
    }
  }

  static Group *get(int);

  static Pool<Group> pool;
};

#endif