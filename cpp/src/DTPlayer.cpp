#include <iomanip>
#include "DTPlayer.h"
#include "go.h"

using namespace std;

Pool<DTNode> DTPlayer::POOL("DT");

DTPlayer::DTPlayer(const std::string& pdfile, int depth) : Player(nullptr)
{
    _best = nullptr;
    _depth = depth;
    _net = new DTResnet(pdfile);
}

// alpha -10000
// beta 10000
float DTPlayer::alphabeta(DTNode *node, int depth, float alpha, float beta)
{
    if (depth == 0 || node->children().size() == 0)
    {
        return node->evaluate();
    }

    node->expand(_net);

    for (auto i = node->children().begin(); i != node->children().end(); ++i)
    {
        DTNode *child = *i;
        float val = alphabeta(child, depth - 1, -beta, -alpha);
        if (val >= beta) //cutting
        {
            return beta;
        }
        if (val > alpha)
        {
            alpha = val;
        }
    }

    return alpha;
}

bool DTPlayer::move()
{
    DTNode *root = nullptr;

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
        root->init(go::POSITION, 0.0f);
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
    }

    clock_t start = clock();

    auto children = root->expand(_net);
    if (children.empty())
    {
        root->release();
        return false;
    }
    else
    {
        _best = children[0];
        float alpha = -1000000.0f;
        float beta = 1000000.0f;
        for (size_t i = 1; i < children.size(); ++i)
        {
            DTNode *node = children[i];
            float val = alphabeta(node, _depth - 1, -beta, -alpha);
            if (val > alpha)
            {
                alpha = val;
                _best = children[i];
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
             << " MP:" << DTPlayer::POOL.size()
             << " Q:" << setprecision(2) << alpha
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

void DTPlayer::clear(void)
{
    if (_best != nullptr)
    {
        _best->release();
        _best = nullptr;
    }
}