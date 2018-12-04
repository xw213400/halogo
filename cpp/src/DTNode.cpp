#include "DTNode.h"
#include "go.h"
#include "DTPlayer.h"

using namespace std;

DTNode::DTNode()
{
    _children.reserve(BRANCHES);
    _position = nullptr;
    _evaluate = 0.0f;
}

DTNode::~DTNode(void)
{
}

void DTNode::init(Position *position, float evaluate)
{
    _position = position;
    _evaluate = evaluate;

    _children.clear();
}

const vector<DTNode *> &DTNode::expand(DTResnet *net)
{
    if (_children.empty())
    {
        net->get(_position, _children);
    }

    return _children;
}

void DTNode::release(bool recursive)
{
    if (!go::isTrunk(_position))
    {
        _position->release();
    }

    if (recursive)
    {
        for (auto i = _children.begin(); i != _children.end(); ++i)
        {
            (*i)->release();
        }
    }

    _children.clear();

    DTPlayer::POOL.push(this);
}