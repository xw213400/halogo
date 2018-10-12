#include "RandMove.h"

RandMove::RandMove(float puct) : Policy(puct)
{
}

RandMove::~RandMove(void)
{
}

void RandMove::get(Position *position, std::vector<Position*>& positions)
{
    
}

float RandMove::sim(Position *position)
{
    
}

void RandMove::clear(void)
{
    hash.clear();
}