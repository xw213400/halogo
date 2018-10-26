
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/cc/ops/nn_ops.h"

using namespace std;
using namespace tensorflow;

// Tensor conv5x5(int in_channel, int out_channel)
// {
//     ops::Conv2D conv2d(in_channel, out_channel, 5);
// }

int main(int argc, char *argv[])
{
    Session *session;
    Status status = NewSession(SessionOptions(), &session);

    Tensor a(DT_FLOAT, TensorShape({2, 2}));
    Tensor b(DT_FLOAT, TensorShape({2, 2}));

    cout << "A:" << a.DebugString() << ", B:" << b.DebugString() << endl;

    return 0;
}