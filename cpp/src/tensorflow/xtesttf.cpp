

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"

using namespace std;
using namespace tensorflow;

int main(int argc, char *argv[])
{
    Scope root = Scope::NewRootScope();

    auto a = ops::Placeholder(root, DT_INT32);
    auto b = ops::Variable(root, {1, 2}, DT_FLOAT);
    auto c = ops::Add(root, a, {41, 43});

    ClientSession session(root);
    vector<Tensor> outputs;

    Status s = session.Run({{a, {1, 2}}}, {c}, &outputs);

    if (!s.ok())
    {
        cout << "Sesion run error!" << endl;
    }

    for (size_t i = 0; i != outputs.size(); ++i)
    {
        cout << outputs[i].flat<int>() << endl;
    }

    GraphDef def;
    Status ss = root.ToGraphDef(&def);

    if (!ss.ok())
    {
        cout << "ToGraphDef error!" << endl;
    }

    cout << def.DebugString() << endl;

    return 0;
}