
#include "Resnet.h"

using namespace std;
using namespace tensorflow;

int main(int argc, char *argv[])
{
    Tensor t(DT_FLOAT, {2, 2});
    t.flat<float>()(3) = -1;
    cout << "Test:" << t.flat<float>()(3) << endl;
    cout << t.DebugString() << endl;

    return 0;
}