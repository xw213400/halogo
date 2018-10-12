#ifndef __GROUP_H__
#define __GROUP_H__

#include <vector>
#include <stdint.h>

class Group
{
  public:
    Group();
    ~Group();

    int getLiberty(int8_t *);

    std::vector<int> stones;
    int liberty;
};

#endif