#include <math.h>
#include "MCTSNode.h"
#include "go.h"
#include "MCTSPlayer.h"

using namespace std;

MCTSNode::MCTSNode(void)
{
    positions.reserve(go::LN + 1);
    children.reserve(go::LN + 1);
    parent = nullptr;
    position = nullptr;
    leaves = 0;
}

MCTSNode::~MCTSNode(void)
{
}

void MCTSNode::init(Policy *pPolicy, MCTSNode *pParent, Position *pPosition)
{
    policy = pPolicy;
    parent = pParent;
    position = pPosition;

    positions.clear();
    if (position->passCount() < 2)
    {
        policy->get(position, positions);
    }

    leaves = positions.size();

    MCTSNode *p = parent;
    while (p != nullptr)
    {
        p->leaves += leaves;
        p = p->parent;
    }

    children.clear();
    score = U = 0.f;
    N = 0;
    Q = 0.f;//parent == nullptr ? 0.f : parent->Q;
}

MCTSNode *MCTSNode::select(void)
{
    if (positions.empty())
    {
        float bestScore = -100000000.f;
        MCTSNode *bestNode = nullptr;
        for (auto i = children.begin(); i != children.end(); ++i)
        {
            auto node = *i;
            if (node->leaves > 0 && node->score > bestScore)
            {
                bestScore = node->score;
                bestNode = node;
            }
        }
        if (bestNode == nullptr)
        {
            return nullptr;
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
    auto pos = positions.back();
    positions.pop_back();
    auto node = MCTSPlayer::pool.pop();
    node->init(policy, this, pos);
    children.push_back(node);

    leaves--;
    MCTSNode *p = parent;
    while (p != nullptr)
    {
        p->leaves--;
        p = p->parent;
    }

    return node;
}

float MCTSNode::simulate(void)
{
    if (positions.empty())
    {
        return position->score();
    }
    else
    {
        return policy->sim(positions.back());
    }
}

void MCTSNode::backpropagation(float value)
{
    if (position->getNext() == go::BLACK)
    {
        value = -value;
    }
    ++N;
    float invN = 1.0f / N;
    Q += (value - Q) * invN;
    if (parent != nullptr)
    {
        U = policy->getPUCT() * sqrt(log(parent->getN() + 1) * invN);
        score = Q + U;
    }
}

void MCTSNode::release(bool recursive)
{
    if (!go::isTrunk(position))
    {
        position->release();
    }

    for (auto i = positions.begin(); i != positions.end(); ++i)
    {
        (*i)->release();
    }

    positions.clear();

    MCTSPlayer::pool.push(this);

    for (auto i = children.begin(); i != children.end(); ++i)
    {
        if (recursive)
        {
            (*i)->release();
        }
        else
        {
            (*i)->parent = nullptr;
        }
    }
}