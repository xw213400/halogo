#include <cstring>
#include "Position.h"
#include "go.h"

using namespace std;
using namespace rapidjson;

Position::Position(void)
{
    _board = new int8_t[go::LM];
    memcpy(_board, go::EMPTY_BOARD, sizeof(int8_t) * go::LM);
    _group = static_cast<Group **>(malloc(sizeof(nullptr) * go::LM));
    memset(_group, 0, sizeof(nullptr) * go::LM);
    _vertex = go::PASS;
    _hashCode = 0;
    _ko = 0;
    _next = go::BLACK;
    _parent = nullptr;
    _dirty = true;
}

Position::~Position(void)
{
    delete _board;
    delete _group;
}

Position *Position::move(int v)
{
    if (v == go::PASS)
    {
        Position *pos = go::POSITION_POOL.pop();
        pos->_parent = this;
        pos->_ko = 0;
        pos->_next = -_next;
        pos->_vertex = v;
        pos->_hashCode = _hashCode ^ go::CODE_KO[_ko] ^ go::CODE_SWAP;
        memcpy(pos->_board, _board, sizeof(int8_t) * go::LV);

        return pos;
    }

    if (_board[v] != go::EMPTY || v == _ko)
    {
        return nullptr;
    }

    bool resonable = false;
    int ko = 0;
    int vu = v + go::UP;
    int vd = v + go::DOWN;
    int vl = v + go::LEFT;
    int vr = v + go::RIGHT;

    int8_t cu = _board[vu];
    int8_t cd = _board[vd];
    int8_t cl = _board[vl];
    int8_t cr = _board[vr];

    Group *gs[4];
    int ngs = 0;

    Group *gu = _group[vu];
    if (gu != nullptr)
    {
        gs[ngs++] = gu;
    }

    Group *gd = _group[vd];
    if (gd != nullptr && gd != gu)
    {
        gs[ngs++] = gd;
    }

    Group *gl = _group[vl];
    if (gl != nullptr && gl != gu && gl != gd)
    {
        gs[ngs++] = gl;
    }

    Group *gr = _group[vr];
    if (gr != nullptr && gr != gu && gr != gd && gr != gl)
    {
        gs[ngs++] = gr;
    }

    int8_t ee = -_next;

    bool bNeighborEmpty = cu * cd * cl * cr == 0;
    bool bNeighborNoEnemy = (cu - ee) * (cd - ee) * (cl - ee) * (cr - ee) != 0;
    bool bNeighborNoFriend = (cu + ee) * (cd + ee) * (cl + ee) * (cr + ee) != 0;

    resonable = bNeighborEmpty;

    int nTakes = 0;

    for (int i = 0; i != ngs; ++i)
    {
        Group *g = gs[i];
        int8_t color = g->color(_board);
        if (color == ee)
        {
            int liberty = g->liberty(_board);
            if (liberty == 1)
            {
                for (int s = 0; s != g->n(); ++s)
                {
                    go::FRONTIER[nTakes++] = g->getStone(s);
                }
            }
        }
        else
        {
            if (!resonable)
            {
                int liberty = g->liberty(_board);
                if (liberty > 1)
                {
                    resonable = true;
                }
            }
        }
    }

    if (nTakes > 0)
    {
        if (!bNeighborEmpty && bNeighborNoFriend && nTakes == 1)
        {
            ko = go::FRONTIER[0];
        }
        resonable = true;
    }

    if (!bNeighborEmpty && bNeighborNoEnemy && ngs == 1)
    {
        resonable = false;
    }

    if (!resonable)
    {
        return nullptr;
    }

    uint64_t hashCode = _hashCode;
    if (_next == go::BLACK)
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
            pos = pos->_parent;
        }
    }

    pos = go::POSITION_POOL.pop();
    memcpy(pos->_board, _board, go::LV);
    pos->_board[v] = _next;
    pos->_next = -_next;
    pos->_vertex = v;
    pos->_hashCode = hashCode;
    pos->_ko = ko;

    for (int i = 0; i != nTakes; ++i)
    {
        pos->_board[go::FRONTIER[i]] = go::EMPTY;
    }
    pos->_parent = this;

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
        int8_t c = _board[i];
        colors[c] = true;
        if (c == go::EMPTY && go::FLAGS[i] != go::FLAG)
        {
            go::FLAGS[i] = go::FLAG;
            go::FRONTIER[n++] = i;
        }

        i = m + go::DOWN;
        c = _board[i];
        colors[c] = true;
        if (c == go::EMPTY && go::FLAGS[i] != go::FLAG)
        {
            go::FLAGS[i] = go::FLAG;
            go::FRONTIER[n++] = i;
        }

        i = m + go::LEFT;
        c = _board[i];
        colors[c] = true;
        if (c == go::EMPTY && go::FLAGS[i] != go::FLAG)
        {
            go::FLAGS[i] = go::FLAG;
            go::FRONTIER[n++] = i;
        }

        i = m + go::RIGHT;
        c = _board[i];
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
        int8_t c = _board[v];
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

void Position::resetLiberty()
{
    for (int i = 0; i != go::LN; ++i)
    {
        Group *g = _group[go::COORDS[i]];
        if (g != nullptr)
        {
            g->resetLiberty();
        }
    }
}

void Position::updateGroup(void)
{
    if (!_dirty)
    {
        for (int i = 0; i != go::LN; ++i)
        {
            Group *g = _group[go::COORDS[i]];
            if (g != nullptr)
            {
                g->resetLiberty();
            }
        }
    }
    else if (_parent != nullptr)
    {
        _dirty = false;

        for (int i = 0; i != go::LN; ++i)
        {
            int v = go::COORDS[i];
            Group *g = _parent->_group[v];
            if (g != nullptr)
            {
                _group[v] = g;
                g->reference(1);
                g->resetLiberty();
            }
        }

        if (_vertex != go::PASS)
        {
            Group *g = _group[_vertex] = Group::get(_vertex);

            int n = _vertex + go::UP;
            Group *gg = _group[n];
            if (gg != nullptr)
            {
                if (_board[n] == _parent->_next && gg != g)
                {
                    g->reference(gg->n());
                    g->merge(gg);
                    for (int i = 0; i != gg->n(); ++i)
                    {
                        _group[gg->getStone(i)] = g;
                    }
                    gg->reference(-gg->n());
                }
                else if (_board[n] == go::EMPTY)
                {
                    for (int i = 0; i != gg->n(); ++i)
                    {
                        _group[gg->getStone(i)] = nullptr;
                    }
                    gg->reference(-gg->n());
                }
            }

            n = _vertex + go::DOWN;
            gg = _group[n];
            if (gg != nullptr)
            {
                if (_board[n] == _parent->_next && gg != g)
                {
                    g->reference(gg->n());
                    g->merge(gg);
                    for (int i = 0; i != gg->n(); ++i)
                    {
                        _group[gg->getStone(i)] = g;
                    }
                    gg->reference(-gg->n());
                }
                else if (_board[n] == go::EMPTY)
                {
                    for (int i = 0; i != gg->n(); ++i)
                    {
                        _group[gg->getStone(i)] = nullptr;
                    }
                    gg->reference(-gg->n());
                }
            }

            n = _vertex + go::LEFT;
            gg = _group[n];
            if (gg != nullptr)
            {
                if (_board[n] == _parent->_next && gg != g)
                {
                    g->reference(gg->n());
                    g->merge(gg);
                    for (int i = 0; i != gg->n(); ++i)
                    {
                        _group[gg->getStone(i)] = g;
                    }
                    gg->reference(-gg->n());
                }
                else if (_board[n] == go::EMPTY)
                {
                    for (int i = 0; i != gg->n(); ++i)
                    {
                        _group[gg->getStone(i)] = nullptr;
                    }
                    gg->reference(-gg->n());
                }
            }

            n = _vertex + go::RIGHT;
            gg = _group[n];
            if (gg != nullptr)
            {
                if (_board[n] == _parent->_next && gg != g)
                {
                    g->reference(gg->n());
                    g->merge(gg);
                    for (int i = 0; i != gg->n(); ++i)
                    {
                        _group[gg->getStone(i)] = g;
                    }
                    gg->reference(-gg->n());
                }
                else if (_board[n] == go::EMPTY)
                {
                    for (int i = 0; i != gg->n(); ++i)
                    {
                        _group[gg->getStone(i)] = nullptr;
                    }
                    gg->reference(-gg->n());
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
        Group *g = _group[v];
        if (g != nullptr)
        {
            g->reference(-1);
            g->release();
            _group[v] = nullptr;
        }
    }
    memcpy(_board, go::EMPTY_BOARD, go::LV);
    _vertex = go::PASS;
    _hashCode = 0;
    _ko = 0;
    _next = go::BLACK;
    _parent = nullptr;
    _dirty = false;
}

void Position::release()
{
    for (int i = 0; i != go::LN; ++i)
    {
        int v = go::COORDS[i];
        Group *g = _group[v];
        if (g != nullptr)
        {
            g->reference(-1);
            g->release();
            _group[v] = nullptr;
        }
    }
    _dirty = true;
    go::POSITION_POOL.push(this);
}

int Position::passCount()
{
    int pc = 0;
    if (_vertex == 0)
    {
        ++pc;
    }
    if (_parent != nullptr && _parent->_vertex == 0)
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
        BOARD.PushBack(_board[go::COORDS[i]], allocator);
    }

    JSON.AddMember("board", BOARD, allocator);
    JSON.AddMember("next", _next, allocator);
    JSON.AddMember("ko", _ko, allocator);
    JSON.AddMember("_vertex", _vertex, allocator);

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
            int8_t c = _board[go::COORDS[i * go::N + j]];
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
            int8_t c = _board[v];
            Group *g = _group[v];
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