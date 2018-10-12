#include <math.h>
#include "MCTSNode.h"
#include "go.h"
#include "MCTSPlayer.h"

MCTSNode::MCTSNode(void)
{
    positions.reserve(go::LN);
    children.reserve(go::LN);
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
    if (position->passCount() <= 2)
    {
        policy->get(position, positions);
    }

    leaves = positions.size();

    if (parent != nullptr)
    {
        parent->addLeaf(leaves);
    }

    children.clear();
    score = Q = U = 0;
    N = 0;
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
    subLeaf();

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
        go::POSITION_POOL.push(position);
    }

    for (auto i = positions.begin(); i != positions.end(); ++i)
    {
        go::POSITION_POOL.push(*i);
    }

    positions.clear();

    MCTSPlayer::pool.push(this);

    if (recursive)
    {
        for (auto i = children.begin(); i != children.end(); ++i)
        {
            (*i)->release();
        }
    }
}