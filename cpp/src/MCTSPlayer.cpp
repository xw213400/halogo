#include <iomanip>
#include "MCTSPlayer.h"
#include "go.h"

using namespace std;

Pool<MCTSNode> MCTSPlayer::pool(100000);

MCTSPlayer::MCTSPlayer(float _seconds, Policy *_policy)
{
    seconds = _seconds;
    policy = _policy;
    best = nullptr;
}

bool MCTSPlayer::move()
{
    MCTSNode *root = nullptr;

    if (best != nullptr)
    {
        if (best->getPosition()->getNext() == go::POSITION->getNext())
        {
            if (best->getPosition()->getVertex() == go::POSITION->getVertex())
            {
                root = best;
            }
        }
        else
        {
            auto children = best->getChildren();
            for (auto i = children.begin(); i != children.end(); ++i)
            {
                if ((*i)->getPosition()->getVertex() == go::POSITION->getVertex())
                {
                    root = *i;
                }
                else
                {
                    (*i)->release();
                }
            }
            best->release(false);
            best = nullptr;
        }
    }

    if (root == nullptr)
    {
        root = pool.pop();
        root->init(policy, nullptr, go::POSITION);
    }
    else if (root->getPosition() != go::POSITION)
    {
        go::POSITION_POOL.push(root->getPosition());
        root->setPosition(go::POSITION);
        auto children = root->getChildren();
        for (auto i = children.begin(); i != children.end(); ++i)
        {
            (*i)->getPosition()->setParent(go::POSITION);
        }
        auto positions = root->getPositions();
        for (auto i = positions.begin(); i != positions.end(); ++i)
        {
            (*i)->setParent(go::POSITION);
        }
    }

    clock_t start = clock();
    int dt = 0;
    int sims = 0;

    while (dt < seconds && sims < 10000)
    {
        MCTSNode *currentNode = root->select();

        if (currentNode == nullptr)
        {
            cout << "Leaves is empty!" << endl;
            break;
        }

        currentNode = currentNode->expand();

        float score = currentNode->simulate();

        while (currentNode != nullptr)
        {
            currentNode->backpropagation(score);
            currentNode = currentNode->getParent();
        }

        dt = (clock() - start) / CLOCKS_PER_SEC;
        ++sims;
    }

    auto children = root->getChildren();
    if (children.empty())
    {
        root->release();
        return false;
    }
    else
    {
        best = children[0];
        for (size_t i = 1; i < children.size(); ++i)
        {
            MCTSNode *node = children[i];
            if (node->getN() > best->getN())
            {
                best = node;
            }
        }

        go::POSITION_POOL.push(go::POSITION);
        go::POSITION = best->getPosition();

        for (auto i = children.begin(); i != children.end(); ++i)
        {
            if (*i != best)
            {
                (*i)->release();
            }
        }

        root->release(false);

        auto ji = go::toJI(go::POSITION->getVertex());
        cout << "V:[" << ji.second << "," << ji.first << "] POOL:"
             << go::POSITION_POOL.size() << " Q:" << setprecision(2)
             << best->getQ() << " SIM:" << sims << endl;

        return true;
    }
}

void MCTSPlayer::clear(void) {
    policy->clear();
    if (best != nullptr) {
        best->release();
        best = nullptr;
    }
}