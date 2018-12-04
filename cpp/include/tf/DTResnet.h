#ifndef __DTRESNET_H__
#define __DTRESNET_H__

#include "DTNode.h"
#include "tensorflow/core/public/session.h"

class DTNode;

class DTResnet
{
  public:
    DTResnet(const std::string &);

    ~DTResnet(){};

    void get(Position *, std::vector<DTNode *> &);

  private:
    void updateInputBoard(Position *);

    tensorflow::Session *_session;
    tensorflow::GraphDef _def;
    tensorflow::Tensor _input_board;
};

#endif