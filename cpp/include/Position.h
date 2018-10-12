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

    float territory(int);

    float score();

    int passCount();

    void clear();

    rapidjson::Value toJSON(rapidjson::Document::AllocatorType &);

    inline int getVertex() {
      return vertex;
    }

    inline int8_t getNext() {
      return next;
    }

    inline void setParent(Position* pos) {
      parent = pos;
    }

    inline Position* getParent() {
      return parent;
    }

  private:
    int8_t next;
    int ko;
    int8_t *board;
    Group **group;
    Group *mygroup;
    int vertex;
    unsigned long long hash_code;
    Position *parent;
};

#endif