#include <math.h>
#include "MCTSNode.h"
#include "go.h"
#include "MCTSPlayer.h"

using namespace std;

MCTSNode::MCTSNode(void)
{
    _positions.reserve(go::LN + 1);
    _children.reserve(go::LN + 1);
    _parent = nullptr;
    _position = nullptr;
    leaves = 0;
}

MCTSNode::~MCTSNode(void)
{
}

void MCTSNode::init(Policy *policy, MCTSNode *pParent, Position *position)
{
    _policy = policy;
    _parent = pParent;
    _position = position;

    _positions.clear();
    if (_position->passCount() < 2)
    {
        _positions.push_back(_position->move(go::PASS));
        policy->get(_position, _positions);
    }

    leaves = _positions.size();

    MCTSNode *p = _parent;
    while (p != nullptr)
    {
        p->leaves += leaves;
        p = p->_parent;
    }

    _children.clear();
    score = U = 0.f;
    N = 0;
    Q = 0.f;
}

MCTSNode *MCTSNode::select(void)
{
    if (_positions.empty())
    {
        float bestScore = -100000000.f;
        MCTSNode *bestNode = nullptr;
        for (auto i = _children.begin(); i != _children.end(); ++i)
        {
            auto node = *i;
            if (node->score > bestScore)
            {
                bestScore = node->score;
                bestNode = node;
            }
        }
        if (bestNode == nullptr)
        {
            return this;
        }
        else
        {
            return bestNode->select();
        }
    }
    else
    {
        return this;
    }
}

MCTSNode *MCTSNode::expand(void)
{
    if (_positions.empty())
    {
        return this;
    }

    auto pos = _positions.back();
    _positions.pop_back();
    auto node = MCTSPlayer::POOL.pop();
    node->init(_policy, this, pos);
    _children.push_back(node);

    leaves--;
    MCTSNode *p = _parent;
    while (p != nullptr)
    {
        p->leaves--;
        p = p->_parent;
    }

    return node;
}

float MCTSNode::simulate(void)
{
    if (_positions.empty())
    {
        return _position->score();
    }
    else
    {
        return _policy->sim(_positions.back());
    }
}

void MCTSNode::backpropagation(float value)
{
    value -= go::KOMI;
    if (_position->next() == go::BLACK)
    {
        value = -value;
    }
    ++N;
    float invN = 1.0f / N;
    value = value > 0.f ? 1.f : -1.f;
    Q += (value - Q) * invN;
    if (_parent != nullptr)
    {
        U = _policy->PUCT() * sqrt(log(_parent->getN() + 1) * invN);
        score = Q + U;
    }
}

void MCTSNode::release(bool recursive)
{
    if (!go::isTrunk(_position))
    {
        _position->release();
    }

    for (auto i = _positions.begin(); i != _positions.end(); ++i)
    {
        (*i)->release();
    }

    _positions.clear();

    MCTSPlayer::POOL.push(this);

    for (auto i = _children.begin(); i != _children.end(); ++i)
    {
        if (recursive)
        {
            (*i)->release();
        }
        else
        {
            (*i)->_parent = nullptr;
        }
    }
}