#ifndef __DTRESNET_H__
#define __DTRESNET_H__

#include "DTNode.h"
#include "tensorflow/core/public/session.h"
#include "Moves.h"


class DTResnet
{
public:
  DTResnet(const std::string &);

  ~DTResnet(){};

  void get(Position *, Moves<DTNode> &);

private:
  void updateInputBoard(Position *);

  tensorflow::Session *_session;
  tensorflow::GraphDef _def;
  tensorflow::Tensor _input_board;
};

#endif