#ifndef __GROUP_H__
#define __GROUP_H__

#include <vector>

class Group
{
  public:
    Group();
    ~Group();

    int getLiberty(unsigned char *);

    std::vector<int> stones;
    int liberty;
};

#endif