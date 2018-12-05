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
        Moves<DTNode> &ns = _nodes[i];
        ns.resize(branches);
    }
}

// alpha -10000
// beta 10000
float DTPlayer::alphabeta(DTNode *node, int depth, float alpha, float beta)
{
    if (depth == 0)
    {
        return _depth % 2 == 0 ? -node->evaluate() : node->evaluate();
    }

    Moves<DTNode> &ns = _nodes[depth];
    _net->get(node->position(), ns);

    for (size_t i = 0; i != ns.size(); ++i)
    {
        DTNode *child = ns.get(i);

        float val = -alphabeta(child, depth - 1, -beta, -alpha);

        if (val >= beta) //cutting
        {
            return beta;
        }
        if (val > alpha)
        {
            alpha = val;
        }
    }

    ns.clear();

    return alpha;
}

bool DTPlayer::move()
{
    int depth = _depth;

    clock_t start = clock();

    Moves<DTNode> &ns = _nodes[--depth];
    _net->get(go::POSITION, ns);

    DTNode *best = nullptr;
    float alpha = -1000000.0f;
    for (size_t i = 0; i < ns.size(); ++i)
    {
        DTNode *node = ns.get(i);
        float val = -alphabeta(node, depth - 1, -1000000.0f, -alpha);
        auto ji = go::toJI(node->position()->vertex());
        cout << setprecision(4) << val << " " << " V:[" << ji.second << "," << ji.first << "]" << endl;
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

    ns.clear();
    DTNode::Release();

    return true;
}

void DTPlayer::clear(void)
{
}