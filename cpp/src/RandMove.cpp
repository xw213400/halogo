#include <algorithm>
#include <chrono>
#include "RandMove.h"
#include "go.h"

using namespace std;

RandMove::RandMove(float puct) : Policy(puct)
{
}

RandMove::~RandMove(void)
{
}

void RandMove::get(Position *position, vector<Position *> &positions)
{
    positions.push_back(position->move(go::PASS));
    position->getChildren(positions);
    auto seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(positions.begin() + 1, positions.end(), default_random_engine(seed));
}

float RandMove::sim(Position *position)
{
    auto it = hash.find(position->hashCode());
    if (it != hash.end())
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

        if (n >= 2)
        {
            int i = rand() % n;
            pos = positions[i];
            for (auto it = positions.begin(); it != positions.end(); ++it)
            {
                Position *p = *it;
                if (p != pos)
                {
                    p->release();
                }
            }
        }
        else if (n == 1)
        {
            pos = positions[0];
        }
        else
        {
            pos = pos->move(go::PASS);
        }
    }
    
    float score = pos->score();
    while (pos != position)
    {
        pos->release();
        pos = pos->getParent();
    }

    hash.insert(make_pair(position->hashCode(), score));

    return score;
}

void RandMove::clear(void)
{
    hash.clear();
}