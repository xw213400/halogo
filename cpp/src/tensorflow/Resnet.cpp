
#include "go.h"
#include "tf/Resnet.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/platform/env.h"

using namespace std;
using namespace tensorflow;

bool comp(const pair<float, int> &a, const pair<float, int> &b)
{
    return a.first > b.first;
}

Tensor Resnet::INPUT_BOARD(DT_FLOAT, {1, 1, go::N, go::N});

Resnet::Resnet(float puct, const std::string &pdfile) : Policy(puct)
{
    Status s = NewSession(SessionOptions(), &_session);

    if (!s.ok())
    {
        cerr << "NewSession Error!" << endl;
    }

    s = ReadBinaryProto(Env::Default(), pdfile, &_def);

    if (!s.ok())
    {
        cerr << "ReadBinaryProto Error!" << endl;
    }

    s = _session->Create(_def);

    if (!s.ok())
    {
        cerr << "Create Graph Error!" << endl;
    }
}

void Resnet::get(Position *position, std::vector<Position *> &positions)
{
    updateInputBoard(position);
    std::vector<Tensor> outputs;

    TF_CHECK_OK(_session->Run({{"0:0", INPUT_BOARD}}, {"add_7:0"}, {}, &outputs));

    position->updateGroup();

    auto cs = outputs[0].flat<float>().data();

    vector<pair<float, int>> datas(go::LN + 1);

    for (int i = 0; i <= go::LN; ++i)
    {
        datas[i] = make_pair(cs[i], i);
    }

    sort(datas.begin(), datas.end(), comp);

    positions.push_back(position->move(go::PASS));
    for (int i = 0; i <= go::LN; ++i)
    {
        if (datas[i].second != go::LN)
        {
            int v = go::COORDS[datas[i].second];
            Position *pos = position->move(v);
            if (pos != nullptr)
            {
                positions.push_back(pos);
            }
        }
    }
}

float Resnet::sim(Position *position)
{
    auto it = _hash.find(position->hashCode());
    if (it != _hash.end())
    {
        return it->second;
    }

    Position *pos = position;

    while (pos->passCount() < 2)
    {
        updateInputBoard(position);
        std::vector<Tensor> outputs;

        TF_CHECK_OK(_session->Run({{"0:0", INPUT_BOARD}}, {"add_7:0"}, {}, &outputs));

        position->updateGroup();

        auto cs = outputs[0].flat<float>().data();

        vector<pair<float, int>> datas(go::LN + 1);

        for (int i = 0; i <= go::LN; ++i)
        {
            datas[i] = make_pair(cs[i], i);
        }

        sort(datas.begin(), datas.end(), comp);

        Position *pp = nullptr;
        for (int i = 0; i <= go::LN; ++i)
        {
            if (datas[i].second != go::LN)
            {
                int v = go::COORDS[datas[i].second];
                pp = pos->move(v);
                if (pp != nullptr)
                {
                    pos = pp;
                    break;
                }
            }
        }

        if (pp == nullptr)
        {
            pos = pos->move(go::PASS);
        }
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
            INPUT_BOARD.flat<float>()(i) = 2.0f;
        }
        else
        {
            INPUT_BOARD.flat<float>()(i) = position->boardi(v) * position->next();
        }
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
            int8_t c = INPUT_BOARD.flat<float>()(idx);
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