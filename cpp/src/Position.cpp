#include <cstring>
#include "Position.h"
#include "go.h"

using namespace std;
using namespace rapidjson;

Position::Position(void)
{
    board = new int8_t[go::LM];
    memcpy(board, go::EMPTY_BOARD, sizeof(int8_t) * go::LM);
    group = static_cast<Group **>(malloc(sizeof(nullptr) * go::LM));
    memset(group, 0, sizeof(nullptr) * go::LM);
    vertex = go::PASS;
    _hashCode = 0;
    _ko = 0;
    next = go::BLACK;
    parent = nullptr;
    groupDirty = true;
}

Position::~Position(void)
{
    delete board;
    delete group;
}

Position *Position::move(int v)
{
    if (v == go::PASS)
    {
        Position *pos = go::POSITION_POOL.pop();
        pos->parent = this;
        pos->_ko = 0;
        pos->next = -next;
        pos->vertex = v;
        pos->_hashCode = _hashCode ^ go::CODE_KO[_ko] ^ go::CODE_SWAP;
        memcpy(pos->board, board, sizeof(int8_t) * go::LV);

        return pos;
    }

    if (board[v] != go::EMPTY || v == _ko)
    {
        return nullptr;
    }

    bool resonable = false;
    int ko = 0;
    int vu = v + go::UP;
    int vd = v + go::DOWN;
    int vl = v + go::LEFT;
    int vr = v + go::RIGHT;

    int8_t cu = board[vu];
    int8_t cd = board[vd];
    int8_t cl = board[vl];
    int8_t cr = board[vr];

    Group *gs[4];
    int ngs = 0;

    Group *gu = group[vu];
    if (gu != nullptr)
    {
        gs[ngs++] = gu;
    }

    Group *gd = group[vd];
    if (gd != nullptr && gd != gu)
    {
        gs[ngs++] = gd;
    }

    Group *gl = group[vl];
    if (gl != nullptr && gl != gu && gl != gd)
    {
        gs[ngs++] = gl;
    }

    Group *gr = group[vr];
    if (gr != nullptr && gr != gu && gr != gd && gr != gl)
    {
        gs[ngs++] = gr;
    }

    int8_t ee = -next;

    if (cu * cd * cl * cr == 0)
    {
        resonable = true;
    }

    int nTakes = 0;

    for (int i = 0; i != ngs; ++i)
    {
        Group *g = gs[i];
        int8_t color = board[g->stones[0]];
        if (color == ee)
        {
            int liberty = g->getLiberty(board);
            if (liberty == 1)
            {
                for (auto it = g->stones.begin(); it != g->stones.end(); ++it)
                {
                    go::FRONTIER[nTakes++] = *it;
                }
            }
        }
        else
        {
            if (!resonable)
            {
                int liberty = g->getLiberty(board);
                if (liberty > 1)
                {
                    resonable = true;
                }
            }
        }
    }

    if (nTakes > 0)
    {
        if (!resonable && nTakes == 1 && (cu + ee) * (cd + ee) * (cl + ee) * (cr + ee) != 0)
        {
            ko = go::FRONTIER[0];
        }
        resonable = true;
    }

    if ((cu - ee) * (cd - ee) * (cl - ee) * (cr - ee) != 0 && ngs == 1)
    {
        resonable = false;
    }

    if (!resonable)
    {
        return nullptr;
    }

    uint64_t hashCode = _hashCode;
    if (next == go::BLACK)
    {
        hashCode ^= go::CODE_BLACK[v] ^ go::CODE_KO[_ko] ^ go::CODE_SWAP ^ go::CODE_KO[ko];
    }
    else
    {
        hashCode ^= go::CODE_WHITE[v] ^ go::CODE_KO[_ko] ^ go::CODE_SWAP ^ go::CODE_KO[ko];
    }

    uint64_t *codes = ee == go::BLACK ? go::CODE_BLACK : go::CODE_WHITE;
    for (int i = 0; i != nTakes; ++i)
    {
        hashCode ^= codes[go::FRONTIER[i]];
    }

    Position *pos = this;
    while (pos != nullptr)
    {
        if (pos->_hashCode == hashCode)
        {
            return nullptr;
        }
        else
        {
            pos = pos->parent;
        }
    }

    pos = go::POSITION_POOL.pop();
    memcpy(pos->board, board, go::LV);
    pos->board[v] = next;
    pos->next = -next;
    pos->vertex = v;
    pos->_hashCode = hashCode;
    pos->_ko = ko;

    for (int i = 0; i != nTakes; ++i)
    {
        pos->board[go::FRONTIER[i]] = go::EMPTY;
    }
    pos->parent = this;

    return pos;
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

void Position::updateGroup(void)
{
    if (parent != nullptr && groupDirty)
    {
        groupDirty = false;

        for (int i = 0; i != go::LN; ++i)
        {
            int v = go::COORDS[i];
            Group *g = parent->group[v];
            if (g != nullptr)
            {
                group[v] = g;
                g->rc++;
                g->liberty = -1;
            }
        }

        if (vertex != go::PASS)
        {
            Group *g = group[vertex] = Group::get(vertex);

            int n = vertex + go::UP;
            Group *gg = group[n];
            if (gg != nullptr)
            {
                if (board[n] == parent->next && gg != g)
                {
                    g->rc += gg->stones.size();
                    for (auto it = gg->stones.begin(); it != gg->stones.end(); ++it)
                    {
                        int s = *it;
                        group[s] = g;
                        g->stones.push_back(s);
                    }
                    gg->rc -= gg->stones.size();
                }
                else if (board[n] == go::EMPTY)
                {
                    for (auto it = gg->stones.begin(); it != gg->stones.end(); ++it)
                    {
                        group[*it] = nullptr;
                    }
                    gg->rc -= gg->stones.size();
                }
            }

            n = vertex + go::DOWN;
            gg = group[n];
            if (gg != nullptr)
            {
                if (board[n] == parent->next && gg != g)
                {
                    g->rc += gg->stones.size();
                    for (auto it = gg->stones.begin(); it != gg->stones.end(); ++it)
                    {
                        int s = *it;
                        group[s] = g;
                        g->stones.push_back(s);
                    }
                    gg->rc -= gg->stones.size();
                }
                else if (board[n] == go::EMPTY)
                {
                    for (auto it = gg->stones.begin(); it != gg->stones.end(); ++it)
                    {
                        group[*it] = nullptr;
                    }
                    gg->rc -= gg->stones.size();
                }
            }

            n = vertex + go::LEFT;
            gg = group[n];
            if (gg != nullptr)
            {
                if (board[n] == parent->next && gg != g)
                {
                    g->rc += gg->stones.size();
                    for (auto it = gg->stones.begin(); it != gg->stones.end(); ++it)
                    {
                        int s = *it;
                        group[s] = g;
                        g->stones.push_back(s);
                    }
                    gg->rc -= gg->stones.size();
                }
                else if (board[n] == go::EMPTY)
                {
                    for (auto it = gg->stones.begin(); it != gg->stones.end(); ++it)
                    {
                        group[*it] = nullptr;
                    }
                    gg->rc -= gg->stones.size();
                }
            }

            n = vertex + go::RIGHT;
            gg = group[n];
            if (gg != nullptr)
            {
                if (board[n] == parent->next && gg != g)
                {
                    g->rc += gg->stones.size();
                    for (auto it = gg->stones.begin(); it != gg->stones.end(); ++it)
                    {
                        int s = *it;
                        group[s] = g;
                        g->stones.push_back(s);
                    }
                    gg->rc -= gg->stones.size();
                }
                else if (board[n] == go::EMPTY)
                {
                    for (auto it = gg->stones.begin(); it != gg->stones.end(); ++it)
                    {
                        group[*it] = nullptr;
                    }
                    gg->rc -= gg->stones.size();
                }
            }
        }
    }
}

void Position::getChildren(vector<Position *> &positions)
{
    updateGroup();

    for (int i = 0; i != go::LN; ++i)
    {
        int v = go::COORDS[i];
        Position *pos = move(v);
        if (pos != nullptr)
        {
            positions.push_back(pos);
        }
    }
}

void Position::clear()
{
    for (int i = 0; i != go::LN; ++i)
    {
        int v = go::COORDS[i];
        Group *g = group[v];
        if (g != nullptr)
        {
            g->rc--;
            g->release();
            group[v] = nullptr;
        }
    }
    memcpy(board, go::EMPTY_BOARD, go::LV);
    vertex = go::PASS;
    _hashCode = 0;
    _ko = 0;
    next = go::BLACK;
    parent = nullptr;
    groupDirty = true;
}

void Position::release()
{
    for (int i = 0; i != go::LN; ++i)
    {
        int v = go::COORDS[i];
        Group *g = group[v];
        if (g != nullptr)
        {
            g->rc--;
            g->release();
            group[v] = nullptr;
        }
    }
    groupDirty = true;
    go::POSITION_POOL.push(this);
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
    JSON.AddMember("ko", _ko, allocator);
    JSON.AddMember("vertex", vertex, allocator);

    return JSON;
}

void Position::debug()
{
    int i = go::N;
    string s = "\n";
    while (i > 0)
    {
        s += (i < 10 ? "0" : "") + to_string(i) + " ";
        i--;
        int j = 0;
        while (j < go::N)
        {
            int8_t c = board[go::COORDS[i * go::N + j]];
            j += 1;
            if (c == go::BLACK)
            {
                s += "X ";
            }
            else if (c == go::WHITE)
            {
                s += "O ";
            }
            else
            {
                s += "+ ";
            }
        }
        s += "\n";
    }

    string x_ = "ABCDEFGHJKLMNOPQRSTYVWYZ";
    s += "   ";
    while (i < go::N)
    {
        s += x_.substr(i, 1) + " ";
        i++;
    }

    cout << s << endl;
}

void Position::debugGroup()
{
    int i = go::N;
    string s = "\n";
    while (i > 0)
    {
        s += (i < 10 ? "0" : "") + to_string(i) + " ";
        i--;
        int j = 0;
        while (j < go::N)
        {
            int v = go::COORDS[i * go::N + j];
            int8_t c = board[v];
            Group *g = group[v];
            j += 1;
            if (c == go::BLACK && g != nullptr)
            {
                s += "X ";
            }
            else if (c == go::WHITE && g != nullptr)
            {
                s += "O ";
            }
            else
            {
                s += "+ ";
            }
        }
        s += "\n";
    }

    string x_ = "ABCDEFGHJKLMNOPQRSTYVWYZ";
    s += "   ";
    while (i < go::N)
    {
        s += x_.substr(i, 1) + " ";
        i++;
    }

    cout << s << endl;
}