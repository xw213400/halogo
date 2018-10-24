#include <algorithm>
#include <chrono>
#include "RandMove.h"
#include "go.h"

using namespace std;

RandMove::RandMove(float puct) : Policy(puct)
{
}

void RandMove::get(Position *position, vector<Position *> &positions)
{
    position->getChildren(positions);
    for (auto i = positions.size()-1; i > 1; --i)
    {
        swap(positions[i], positions[rand() % i + 1]);
    }
}

float RandMove::sim(Position *position)
{
    auto it = _hash.find(position->hashCode());
    if (it != _hash.end())
    {
        return it->second;
    }

    Position *pos = position;

    vector<Position *> positions(go::LN);

    while (pos->passCount() < 2)
    {
        positions.clear();

        pos->getChildren(positions);

        size_t n = positions.size();

        Position *pp = nullptr;

        if (n >= 2)
        {
            int i = rand() % n;
            pp = positions[i];
            for (auto it = positions.begin(); it != positions.end(); ++it)
            {
                Position *p = *it;
                if (p != pp)
                {
                    p->release();
                }
            }
        }
        else if (n == 1)
        {
            pp = positions[0];
        }
        else
        {
            pp = pos->move(go::PASS);
        }

        pos = pp;
    }

    float score = pos->score();
    while (pos != position)
    {
        pos->release();
        pos = pos->parent();
    }

    _hash.insert(make_pair(position->hashCode(), score));

    return score;
}

void RandMove::clear(void)
{
    _hash.clear();
}