
#include "go.h"
#include "tf/Resnet.h"

using namespace std;
using namespace tensorflow;

bool compareScore(const pair<float, int> &a, const pair<float, int> &b)
{
    return a.first > b.first;
}

Resnet::Resnet(NNParam* param) : Policy(param->puct), _input_board(DT_FLOAT, {1, 2, go::N, go::N})
{
    Status s = NewSession(SessionOptions(), &_session);

    if (!s.ok())
    {
        cerr << "NewSession Error!" << endl;
    }

    s = ReadBinaryProto(Env::Default(), param->pdfile, &_def);

    if (!s.ok())
    {
        cerr << "ReadBinaryProto Error!" << endl;
    }

    s = _session->Create(_def);

    if (!s.ok())
    {
        cerr << "Create Graph Error!" << endl;
    }

    _param = param;
}

void Resnet::get(Position *position, std::vector<Position *> &positions)
{
    updateInputBoard(position);
    std::vector<Tensor> outputs;

    TF_CHECK_OK(_session->Run({{"0:0", _input_board}}, {"add_7:0"}, {}, &outputs));

    position->updateGroup();

    auto cs = outputs[0].flat<float>().data();

    vector<pair<float, int>> datas(go::LN + 1);

    for (int i = 0; i <= go::LN; ++i)
    {
        datas[i] = make_pair(cs[i], i);
    }

    sort(datas.begin(), datas.end(), compareScore);

    for (int i = 0; i <= go::LN; ++i)
    {
        if (datas[i].second != go::LN)
        {
            int v = go::COORDS[datas[i].second];
            Position *pos = position->move(v);
            if (pos != nullptr)
            {
                positions.push_back(pos);
                if (positions.size() >= _param->branches)
                {
                    break;
                }
            }
        }
    }

    if (positions.size() < 10)
    {
        positions.push_back(position->move(go::PASS));
    }
    reverse(positions.begin(), positions.end());
}

float Resnet::sim(Position *position)
{
    auto it = _hash.find(position->hashCode());
    if (it != _hash.end())
    {
        return it->second;
    }

    Position *pos = position;
    vector<Position *> positions(go::LN);
    int steps = pos->getSteps();
    int simstep = 0;

    Position *pp = nullptr;
    Position *ppp = nullptr;

    while (pos->passCount() < 2)
    {
        if (steps <= go::LN && simstep < _param->simstep)
        {
            simstep++;

            updateInputBoard(position);
            std::vector<Tensor> outputs;

            TF_CHECK_OK(_session->Run({{"0:0", _input_board}}, {"add_7:0"}, {}, &outputs));

            position->updateGroup();

            auto cs = outputs[0].flat<float>().data();

            vector<pair<float, int>> datas(go::LN + 1);

            for (int i = 0; i <= go::LN; ++i)
            {
                datas[i] = make_pair(cs[i], i);
            }

            sort(datas.begin(), datas.end(), compareScore);

            pp = ppp = nullptr;
            pos->updateGroup();

            for (int i = 0; i <= go::LN; ++i)
            {
                if (datas[i].second != go::LN)
                {
                    int v = go::COORDS[datas[i].second];
                    pp = pos->move(v);
                    if (pp != nullptr)
                    {
                        if (ppp == nullptr)
                        {
                            ppp = pp;
                        }
                        if ((rand() % 10000) * 0.0001 <= _param->simrand)
                        {
                            pos = pp;
                            break;
                        }
                        else
                        {
                            if (pp != ppp)
                            {
                                pp->release();
                            }
                        }
                    }
                }
            }

            if (ppp == nullptr)
            {
                pos = pos->move(go::PASS);
            }
            else
            {
                if (pos != pp)
                {
                    pos = ppp;
                }
                if (pos != ppp)
                {
                    ppp->release();
                }
            }
        }
        else
        {
            positions.clear();

            pos->getChildren(positions);

            size_t n = positions.size();

            pp = nullptr;

            if (n >= 2)
            {
                int i = rand() % n;
                pp = positions[i];
                for (auto it = positions.begin(); it != positions.end(); ++it)
                {
                    ppp = *it;
                    if (ppp != pp)
                    {
                        ppp->release();
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

        steps++;
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

void Resnet::clear()
{
    _hash.clear();
}

void Resnet::updateInputBoard(Position *position)
{
    for (int i = 0; i != go::LN; ++i)
    {
        int v = go::COORDS[i];

        if (v == position->ko())
        {
            _input_board.flat<float>()(i) = -1.0f;
        }
        else
        {
            _input_board.flat<float>()(i) = 1.0f;
        }
        _input_board.flat<float>()(i + go::LN) = position->boardi(v) * position->next();
    }
}

void Resnet::debugInput()
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
            int idx = i * go::N + j;
            int8_t c = _input_board.flat<float>()(idx);
            j += 1;
            if (c == go::BLACK)
            {
                s += "X ";
            }
            else if (c == go::WHITE)
            {
                s += "O ";
            }
            else if (c == go::WALL) //KO
            {
                s += "# ";
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