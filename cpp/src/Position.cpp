#include <cstring>
#include "Position.h"
#include "go.h"

using namespace rapidjson;

Position::Position(void)
{
    board = new int8_t[go::LM];
    memcpy(board, go::EMPTY_BOARD, go::LM);
    group = static_cast<Group **>(malloc(sizeof(nullptr) * go::LM));
    memset(group, 0, sizeof(nullptr) * go::LM);
    mygroup = new Group();
    vertex = go::PASS;
    hash_code = 0;
    ko = 0;
    next = go::BLACK;
    parent = nullptr;
}

Position::~Position(void)
{
    delete board;
    delete group;
    delete mygroup;
}

Position *Position::move(int v)
{
    if (v == go::PASS)
    {
        Position *pos = go::POSITION_POOL.pop();
    }
}

float Position::territory(int v)
{
    go::FRONTIER[0] = v;
    go::FLAGS[v] = go::FLAG;
    int n = 1;
    int empties = 0;

    // flags of EMPTY, BLACK, WALL, WHITE
    bool colors[4] = {false, false, false, false};

    while (n > 0)
    {
        n--;
        int m = go::FRONTIER[n];
        empties++;

        int i = m + go::UP;
        int8_t c = board[i];
        colors[c] = true;
        if (c == go::EMPTY && go::FLAGS[i] != go::FLAG)
        {
            go::FLAGS[i] = go::FLAG;
            go::FRONTIER[n++] = i;
        }

        i = m + go::DOWN;
        c = board[i];
        colors[c] = true;
        if (c == go::EMPTY && go::FLAGS[i] != go::FLAG)
        {
            go::FLAGS[i] = go::FLAG;
            go::FRONTIER[n++] = i;
        }

        i = m + go::LEFT;
        c = board[i];
        colors[c] = true;
        if (c == go::EMPTY && go::FLAGS[i] != go::FLAG)
        {
            go::FLAGS[i] = go::FLAG;
            go::FRONTIER[n++] = i;
        }

        i = m + go::RIGHT;
        c = board[i];
        colors[c] = true;
        if (c == go::EMPTY && go::FLAGS[i] != go::FLAG)
        {
            go::FLAGS[i] = go::FLAG;
            go::FRONTIER[n++] = i;
        }
    }

    if (colors[go::BLACK])
    {
        if (colors[go::WHITE])
        {
            return 0.f;
        }
        else
        {
            return empties;
        }
    }
    else
    {
        if (colors[go::WHITE])
        {
            return -empties;
        }
        else
        {
            return 0.f;
        }
    }
}

float Position::score()
{
    go::FLAG++;
    float score = 0.f;
    for (int i = 0; i != go::LN; ++i)
    {
        int v = go::COORDS[i];
        int8_t c = board[v];
        if (c == go::EMPTY && go::FLAGS[v] != go::FLAG)
        {
            score += territory(v);
        }
        else
        {
            score += c;
        }
    }
    return score;
}

void Position::clear()
{
    memcpy(board, go::EMPTY_BOARD, go::LM);
    memset(group, 0, sizeof(nullptr) * go::LM);
    vertex = go::PASS;
    hash_code = 0;
    ko = 0;
    next = go::BLACK;
    parent = nullptr;
}

int Position::passCount()
{
    int pc = 0;
    if (vertex == 0)
    {
        ++pc;
    }
    if (parent != nullptr && parent->vertex == 0)
    {
        ++pc;
    }
    return pc;
}

Value Position::toJSON(Document::AllocatorType &allocator)
{
    Value JSON(kObjectType);

    Value BOARD(kArrayType);

    for (int i = 0; i != go::LN; ++i)
    {
        BOARD.PushBack(board[go::COORDS[i]], allocator);
    }

    JSON.AddMember("board", BOARD, allocator);
    JSON.AddMember("next", next, allocator);
    JSON.AddMember("ko", ko, allocator);
    JSON.AddMember("vertex", vertex, allocator);

    return JSON;
}