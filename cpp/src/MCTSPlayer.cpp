#include <iomanip>
#include "MCTSPlayer.h"
#include "go.h"

using namespace std;

Pool<MCTSNode> MCTSPlayer::POOL("MCTS");

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
        if (best->position()->next() == go::POSITION->next())
        {
            if (best->position()->vertex() == go::POSITION->vertex())
            {
                root = best;
            }
        }
        else
        {
            auto children = best->children();
            for (auto i = children.begin(); i != children.end(); ++i)
            {
                if ((*i)->position()->vertex() == go::POSITION->vertex())
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
        root = POOL.pop();
        root->init(policy, nullptr, go::POSITION);
    }
    else if (root->position() != go::POSITION)
    {
        root->position()->release();
        root->position(go::POSITION);
        auto children = root->children();
        for (auto i = children.begin(); i != children.end(); ++i)
        {
            (*i)->position()->parent(go::POSITION);
        }
        auto positions = root->positions();
        for (auto i = positions.begin(); i != positions.end(); ++i)
        {
            (*i)->parent(go::POSITION);
        }
    }

    clock_t start = clock();
    int dt = 0;
    int sims = 0;

    while (dt < seconds && sims < 10000)
    {
        MCTSNode *currentNode = root->select();

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

    auto children = root->children();
    if (children.empty())
    {
        root->release();
        return false;
    }
    else
    {
        best = children[0];
        // cout << best->text() << endl;
        for (size_t i = 1; i < children.size(); ++i)
        {
            MCTSNode *node = children[i];
            // cout << node->text() << endl;
            if (node->getN() > best->getN())
            {
                best = node;
            }
        }

        int vertex = best->position()->vertex();

        go::POSITION->resetLiberty();
        go::POSITION = go::POSITION->move(vertex);
        go::POSITION->updateGroup();

        // go::POSITION->debug();

        auto ji = go::toJI(go::POSITION->vertex());
        cout << setw(3) << setfill('0') << go::POSITION->getSteps()
             << " V:[" << ji.second << "," << ji.first << "] PP:"
             << go::POSITION_POOL.size() << " GP:" << Group::POOL.size()
             << " MP:" << MCTSPlayer::POOL.size() << " Q:" << setprecision(2)
             << best->getQ() << " SIM:" << sims << endl;

        for (auto i = children.begin(); i != children.end(); ++i)
        {
            if (*i != best)
            {
                (*i)->release();
            }
        }

        root->release(false);

        return true;
    }
}

void MCTSPlayer::clear(void)
{
    policy->clear();
    if (best != nullptr)
    {
        best->release();
        best = nullptr;
    }
}