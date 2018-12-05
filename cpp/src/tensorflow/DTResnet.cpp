#include "go.h"
#include "tf/DTResnet.h"
#include "DTPlayer.h"

using namespace std;
using namespace tensorflow;

bool comp(const pair<float, int> &a, const pair<float, int> &b)
{
    return a.first > b.first;
}

DTResnet::DTResnet(const string &pdfile) : _input_board(DT_FLOAT, {1, 2, go::N, go::N})
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

void DTResnet::get(Position *position, Moves<DTNode> &nodes)
{
    updateInputBoard(position);
    std::vector<Tensor> outputs;

    TF_CHECK_OK(_session->Run({{"0:0", _input_board}}, {"add_7:0"}, {}, &outputs));

    position->updateGroup();

    auto cs = outputs[0].flat<float>().data();

    vector<pair<float, int>> datas(go::LN + 1);

    float passscore = cs[go::LN];

    for (int i = 0; i < go::LN; ++i)
    {
        datas[i] = make_pair(cs[i], i);
    }

    sort(datas.begin(), datas.end(), comp);

    for (int i = 0; i <= go::LN; ++i)
    {
        int v = go::COORDS[datas[i].second];
        Position *pos = position->move(v);
        if (pos != nullptr)
        {
            DTNode *node = nodes.add();
            node->init(pos, datas[i].first);
            if (nodes.full())
            {
                break;
            }
        }
    }

    if (!nodes.full())
    {
        DTNode *node = nodes.add();
        node->init(position->move(go::PASS), passscore);
    }
}

void DTResnet::updateInputBoard(Position *position)
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
