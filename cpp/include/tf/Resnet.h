#ifndef __RESNET_H__
#define __RESNET_H__

#include <unordered_map>
#include "Policy.h"
#include "tensorflow/core/public/session.h"

struct NNParam
{
    float puct = 0.5f;
    std::string pdfile;
    std::size_t branches = 20;
    int simstep = 8;
    float simrand = 0.5f;
};

class Resnet : public Policy
{
  public:
    Resnet(NNParam*);

    virtual ~Resnet(){};

    void get(Position *, std::vector<Position *> &);

    float sim(Position *);

    void clear();

  private:
    void updateInputBoard(Position *);

    void debugInput();

    NNParam* _param;
    
    tensorflow::Session *_session;
    tensorflow::GraphDef _def;
    std::unordered_map<uint64_t, float> _hash;

    tensorflow::Tensor _input_board;
};

#endif