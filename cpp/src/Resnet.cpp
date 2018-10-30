
#include "go.h"
#include "Resnet.h"
#include "tensorflow/cc/ops/array_ops.h"

using namespace std;
using namespace tensorflow;

Output Resnet::conv5x5(Input x, int in_channel, int out_channel)
{
    auto filter = ops::Variable(_scope, {5, 5, in_channel, out_channel}, DT_FLOAT);
    return ops::Conv2D(_scope, x, filter, {1, 1, 1, 1}, "SAME");
}

Output Resnet::residualBlock(Input x, int in_channel, int out_channel)
{
    auto conv = conv5x5(x, in_channel, out_channel);
    auto y = ops::Add(_scope, conv, x);
    auto out = ops::Relu(_scope, y);

    return out;
}

void Resnet::resnet(int num_planes)
{
    Output x = ops::Const<float>(_scope.WithOpName("x"), DT_FLOAT, {1, go::N, go::N, 1});

    auto block0 = conv5x5(x, 1, num_planes);

    auto block1 = residualBlock(block0, num_planes, num_planes);
    auto block2 = residualBlock(block1, num_planes, num_planes);
    auto block3 = residualBlock(block2, num_planes, num_planes);
    auto block4 = residualBlock(block3, num_planes, num_planes);
    auto block5 = residualBlock(block4, num_planes, num_planes);

    auto filter = ops::Variable(_scope, {1, 1, num_planes, 4}, DT_FLOAT);
    auto classifier = ops::Conv2D(_scope, block5, filter, {1, 1, 1, 1}, "VALID");

    // auto shape = ops::Variable(_scope, {go::LN * 4}, DT_FLOAT);
    // auto flat = ops::Reshape(_scope, classifier, shape);

    // auto weight = ops::Variable(_scope, {go::LN * 4, go::LN + 1}, DT_FLOAT);
    // auto fc = ops::MatMul(_scope.WithOpName("y"), flat, weight);
}

Resnet::Resnet(float puct) : Policy(puct), _scope(Scope::NewRootScope())
{
    Status s = NewSession(SessionOptions(), &_session);

    if (!s.ok())
    {
        cerr << "NewSession Error!" << endl;
    }

    resnet();

    s = _scope.ToGraphDef(&_def);

    if (!s.ok())
    {
        cerr << "ToGraphDef Error!" << endl;
    }

    _session->Create(_def);

    _sortArray = new int[go::LN + 1];
}

void Resnet::get(Position *position, std::vector<Position *> &positions)
{
    position->updateInputBoard();
    std::vector<Tensor> outputs;
    _session->Run({{"x", go::INPUT_BOARD}}, {"y"}, {}, &outputs);

    position->updateGroup();

    auto cs = outputs[0].flat<float>().data();

    memcpy(_sortArray, go::COORDS, sizeof(int) * go::LN);
    _sortArray[go::LN] = go::PASS;
    quickSort(cs, 0, go::LN, _sortArray);

    positions.push_back(position->move(go::PASS));
    for (int i = 0; i <= go::LN; ++i)
    {
        float v = _sortArray[i];
        if (v != go::PASS)
        {
            Position *pos = position->move(v);
            if (pos != nullptr)
            {
                positions.push_back(pos);
            }
        }
    }
}

float Resnet::sim(Position *position)
{
    auto it = _hash.find(position->hashCode());
    if (it != _hash.end())
    {
        return it->second;
    }

    Position *pos = position;

    while (pos->passCount() < 2)
    {
        position->updateInputBoard();
        std::vector<Tensor> outputs;
        _session->Run({{"input", go::INPUT_BOARD}}, {"output:0"}, {}, &outputs);

        position->updateGroup();

        auto cs = outputs[0].flat<float>().data();

        memcpy(_sortArray, go::COORDS, sizeof(int) * go::LN);
        _sortArray[go::LN] = go::PASS;
        quickSort(cs, 0, go::LN, _sortArray);

        Position *pp = nullptr;
        for (int i = 0; i <= go::LN; ++i)
        {
            float v = _sortArray[i];
            if (v != go::PASS)
            {
                pp = pos->move(v);
                if (pp != nullptr)
                {
                    pos = pp;
                    break;
                }
            }
        }

        if (pp == nullptr)
        {
            pos = pos->move(go::PASS);
        }
    }

    float score = pos->score();
    while (pos != position)
    {
        pos->release();
        pos = pos->parent();
    }

    _hash.insert(make_pair(position->hashCode(), score));

    return score;
}

void Resnet::clear()
{
    _hash.clear();
}
