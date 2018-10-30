#ifndef __RESNET_H__
#define __RESNET_H__

#include <unordered_map>
#include "Policy.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"

class Resnet : public Policy
{
  public:
    Resnet(float);

    virtual ~Resnet(){};

    void get(Position *, std::vector<Position *> &);

    float sim(Position *);

    void clear();

  private:
    tensorflow::Output conv5x5(tensorflow::Input, int in_channel, int out_channel);

    tensorflow::Output residualBlock(tensorflow::Input, int in_channel, int out_channel);

    void resnet(int num_planes = 30);

    void quickSort(float *arr, int l, int r, int* idx)
    {
        float tempa;
        int tempi;
        int i, j, x;
        if (l < r)
        {
            i = l;
            j = r;
            x = arr[(l + r) / 2];
            while (1)
            {
                while (i <= r && arr[i] < x)
                    i++;
                while (j >= 0 && arr[j] > x)
                    j--;
                if (i >= j)
                    break;
                else
                {
                    tempa = arr[i];
                    arr[i] = arr[j];
                    arr[j] = tempa;
                    tempi = idx[i];
                    idx[i] = idx[j];
                    idx[j] = tempi;
                }
            }
            quickSort(arr, l, i - 1, idx);
            quickSort(arr, j + 1, r, idx);
        }
    }

    tensorflow::Session *_session;
    tensorflow::Scope _scope;
    tensorflow::GraphDef _def;
    std::unordered_map<uint64_t, float> _hash;
    int* _sortArray;
};

#endif