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

  void release();

  rapidjson::Value toJSON(rapidjson::Document::AllocatorType &);

  inline int getVertex()
  {
    return vertex;
  }

  inline int8_t getNext()
  {
    return next;
  }

  inline void setParent(Position *pos)
  {
    parent = pos;
  }

  inline Position *getParent()
  {
    return parent;
  }

  void getChildren(std::vector<Position *> &);

  inline uint64_t hashCode()
  {
    return _hashCode;
  }

  inline int getSteps()
  {
    int step = 0;
    Position *p = parent;
    while (p != nullptr)
    {
      step++;
      p = p->parent;
    }
    return step;
  }

  void updateGroup();

  void debug();

  void debugGroup();

private:
  int8_t next;
  int _ko;
  int8_t *board;
  Group **group;
  int vertex;
  uint64_t _hashCode;
  Position *parent;
  bool groupDirty;
};

#endif