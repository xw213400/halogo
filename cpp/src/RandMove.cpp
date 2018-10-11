#include "RandMove.h"

RandMove::RandMove(float puct) : Policy(puct)
{
}

RandMove::~RandMove(void)
{
}

Position **RandMove::get(Position *position)
{
    
}

float RandMove::sim(Position *position)
{
    
}

void RandMove::clear(void)
{
    hash.clear();
}