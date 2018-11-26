#ifndef __RESNET_H__
#define __RESNET_H__

#include <unordered_map>
#include "Policy.h"
#include "tensorflow/core/public/session.h"

class Resnet : public Policy
{
  public:
    Resnet(float, const std::string &);

    virtual ~Resnet(){};

    void get(Position *, std::vector<Position *> &);

    float sim(Position *);

    void clear();

  private:
    void updateInputBoard(Position *);

    void debugInput();

    tensorflow::Session *_session;
    tensorflow::GraphDef _def;
    std::unordered_map<uint64_t, float> _hash;

    static tensorflow::Tensor INPUT_BOARD;
};

#endif