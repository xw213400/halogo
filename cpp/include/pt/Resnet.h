#ifndef __PT_RESNET_H__
#define __PT_RESNET_H__

#include <unordered_map>
#include "Policy.h"
#include "torch/torch.h"
#include "torch/script.h"

class Resnet : public Policy
{
  public:
    Resnet(float, const std::string&);

    virtual ~Resnet(){};

    void get(Position *, std::vector<Position *> &);

    float sim(Position *);

    void clear();

  private:
    void resnet(int num_planes = 30);

    void quickSort(const std::vector<double>& arr, int l, int r, int* idx);

    void updateInputBoard(Position*);

    void debugInput();

    std::shared_ptr<torch::jit::script::Module> _module;
    std::unordered_map<uint64_t, float> _hash;
    int* _sortArray;

    static torch::Tensor INPUT_BOARD;
};

#endif