#include <iomanip>
#include "MCTSPlayer.h"
#include "go.h"

using namespace std;

Pool<MCTSNode> MCTSPlayer::POOL("MCTS");

MCTSPlayer::MCTSPlayer(Policy *policy, int sims) : Player(policy)
{
    _sims = sims;
    _best = nullptr;
}

bool MCTSPlayer::move()
{
    MCTSNode *root = nullptr;

    if (_best != nullptr)
    {
        if (_best->position()->next() == go::POSITION->next())
        {
            if (_best->position()->vertex() == go::POSITION->vertex())
            {
                root = _best;
            }
        }
        else
        {
            auto children = _best->children();
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
            _best->release(false);
            _best = nullptr;
        }
    }

    if (root == nullptr)
    {
        root = POOL.pop();
        root->init(_policy, nullptr, go::POSITION);
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
    int sims = 0;

    while (sims < _sims)
    {
        MCTSNode *currentNode = root->select();

        currentNode = currentNode->expand();

        float score = currentNode->simulate();

        while (currentNode != nullptr)
        {
            currentNode->backpropagation(score);
            currentNode = currentNode->getParent();
        }

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
        _best = children[0];
        for (size_t i = 1; i < children.size(); ++i)
        {
            MCTSNode *node = children[i];
            if (node->getN() > _best->getN())
            {
                _best = node;
            }
        }

        int vertex = _best->position()->vertex();

        go::POSITION->resetLiberty();
        go::POSITION = go::POSITION->move(vertex);
        go::POSITION->updateGroup();

        int dt = (clock() - start) * 1000 / CLOCKS_PER_SEC;

        auto ji = go::toJI(go::POSITION->vertex());
        cout << setw(3) << setfill('0') << go::POSITION->getSteps()
             << " V:[" << ji.second << "," << ji.first << "] PP:"
             << go::POSITION_POOL.size() << " GP:" << Group::POOL.size()
             << " MP:" << MCTSPlayer::POOL.size()
             << " Q:" << setprecision(2) << _best->getQ()
             << " DT:" << dt << endl;

        for (auto i = children.begin(); i != children.end(); ++i)
        {
            if (*i != _best)
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
    _policy->clear();
    if (_best != nullptr)
    {
        _best->release();
        _best = nullptr;
    }
}