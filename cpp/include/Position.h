#ifndef __POSITION_H__
#define __POSITION_H__

#include "Group.h"
#include "rapidjson/document.h"

class Position
{
  public:
    Position();
    ~Position();

    Position *move(int);

    float score();

    int passCount();

    rapidjson::Value& toJSON();

  private:
    int next;
    int ko;
    unsigned char *board;
    Group **group;
    Group *mygroup;
    int vertex;
    unsigned long long hash_code;
    Position *parent;
};

#endif