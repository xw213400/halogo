#include "MCTSPlayer.h"

MCTSPlayer::MCTSPlayer(float _time, Policy* _policy)
{
    seconds_per_move = _time;
    policy = _policy;
}