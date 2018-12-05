#include "DTNode.h"
#include "go.h"
#include "DTPlayer.h"

using namespace std;

vector<Position*> DTNode::POSITIONS;

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

    POSITIONS.push_back(position);
}

void DTNode::Release()
{
    for (auto i = POSITIONS.begin(); i != POSITIONS.end(); ++i)
    {
        (*i)->release();
    }
    POSITIONS.clear();
}