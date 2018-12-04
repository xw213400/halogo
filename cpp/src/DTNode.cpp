#include "DTNode.h"
#include "go.h"
#include "DTPlayer.h"

using namespace std;

DTNode::DTNode()
{
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
}

void DTNode::release(bool recursive)
{
    if (_position != nullptr && !go::isTrunk(_position))
    {
        _position->release();
    }
}