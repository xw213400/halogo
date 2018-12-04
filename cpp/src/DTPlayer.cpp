#include <iomanip>
#include "DTPlayer.h"
#include "go.h"

using namespace std;

DTPlayer::DTPlayer(const std::string &pdfile, int depth, int branches) : Player(nullptr)
{
    _depth = depth;
    _branches = branches;
    _net = new DTResnet(pdfile);
    
    _nodes.resize(depth);
    for (int i = 0; i != depth; ++i)
    {
        Moves<DTNode>& ns = _nodes[i];
        ns.resize(branches);
    }
}

void DTPlayer::clearNodes(Moves<DTNode>& nodes)
{
    for (size_t i = 0; i != nodes.size(); ++i)
    {
        DTNode *node = nodes.get(i);
        node->release();
    }

    nodes.clear();
}

// alpha -10000
// beta 10000
float DTPlayer::alphabeta(DTNode *node, int depth, float alpha, float beta)
{
    if (depth == 0)
    {
        return node->evaluate();
    }

    Moves<DTNode>& ns = _nodes[depth];
    _net->get(node->position(), ns);

    for (size_t i = 0; i != ns.size(); ++i)
    {
        DTNode *child = ns.get(i);

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

    clearNodes(ns);

    return alpha;
}

bool DTPlayer::move()
{
    int depth = _depth-1;

    Moves<DTNode>& nsroot = _nodes[depth--];

    DTNode *root = nsroot.get(0);
    root->init(go::POSITION, 0.0f);

    clock_t start = clock();

    Moves<DTNode>& ns = _nodes[depth--];
    _net->get(root->position(), ns);

    if (ns.empty())
    {
        clearNodes(nsroot);
        return false;
    }
    else
    {
        DTNode *best = ns.get(0);
        float alpha = -1000000.0f;
        float beta = 1000000.0f;
        for (size_t i = 1; i < ns.size(); ++i)
        {
            DTNode *node = ns.get(i);
            float val = alphabeta(node, depth, -beta, -alpha);
            cout << val << " ";
            if (val > alpha)
            {
                alpha = val;
                best = ns.get(i);
            }
        }
        cout << endl;

        int vertex = best->position()->vertex();

        go::POSITION->resetLiberty();
        go::POSITION = go::POSITION->move(vertex);
        go::POSITION->updateGroup();

        int dt = (clock() - start) * 1000 / CLOCKS_PER_SEC;

        auto ji = go::toJI(go::POSITION->vertex());
        cout << setw(3) << setfill('0') << go::POSITION->getSteps()
             << " V:[" << ji.second << "," << ji.first << "] PP:"
             << go::POSITION_POOL.size() << " GP:" << Group::POOL.size()
             << " Q:" << setprecision(2) << alpha
             << " DT:" << dt << endl;

        clearNodes(ns);
        clearNodes(nsroot);

        return true;
    }
}

void DTPlayer::clear(void)
{
}