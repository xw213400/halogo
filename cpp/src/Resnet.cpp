
#include "go.h"
#include "Resnet.h"

using namespace std;
using namespace tensorflow;

Output Resnet::conv5x5(Input x, int in_channel, int out_channel)
{
    return ops::Conv2D(_scope, x, {5, 5, in_channel, out_channel}, {1, 1, 1, 1}, "SAME");
}

Output Resnet::residualBlock(Input x, int in_channel, int out_channel)
{
    auto conv = conv5x5(x, in_channel, out_channel);
    auto y = ops::Add(_scope, conv, x);
    auto out = ops::Relu(_scope, y);

    return out;
}

GraphDef Resnet::resnet(int num_planes)
{
    auto input = ops::Const<float>(_scope.WithOpName("input"), DT_FLOAT, {1, go::N, go::N, 1});
    auto block0 = conv5x5(input, 1, num_planes);

    auto block1 = residualBlock(block0, num_planes, num_planes);
    auto block2 = residualBlock(block1, num_planes, num_planes);
    auto block3 = residualBlock(block2, num_planes, num_planes);
    auto block4 = residualBlock(block3, num_planes, num_planes);
    auto block5 = residualBlock(block4, num_planes, num_planes);

    auto classifier = ops::Conv2D(_scope, block5, {1, 1, num_planes, 4}, {1, 1, 1, 1}, "VALID");
    auto flat = ops::Reshape(_scope, classifier, {go::LN*4});
    auto weight = ops::Variable(_scope, {go::LN*4, go::LN+1}, DT_FLOAT);
    auto fc = ops::MatMul(_scope.WithOpName("output"), flat, weight);

    GraphDef def;
    _scope.ToGraphDef(&def);

    return def;
}

Resnet::Resnet(float puct) : Policy(puct), _scope(Scope::NewRootScope())
{
    Status status = NewSession(SessionOptions(), &_session);

    GraphDef def = resnet();
    _session->Create(def);
}

void Resnet::get(Position *position, std::vector<Position *> &positions) {
    std::vector<Tensor> outputs;
    _session->Run({{"input", go::INPUT_BOARD}},  {"output:0"}, {}, &outputs);
}

float Resnet::sim(Position *position) {

}

void Resnet::clear()
{
    _hash.clear();
}
