#include "MCTSNode.h"
#include "go.h"

MCTSNode::MCTSNode(void):positions(go::LN), children(go::LN)
{
    parent = nullptr;
    position = nullptr;
    leaves = 0;
}

MCTSNode::~MCTSNode(void)
{
}

void MCTSNode::init(MCTSNode* pParent, Position* pPosition)
{
    parent = pParent;
    position = pPosition;

    
}

void MCTSNode::release(bool recursive=true)
{

}