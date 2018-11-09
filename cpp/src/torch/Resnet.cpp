#include "pt/Resnet.h"
#include <torch/script.h>
#include "go.h"

using namespace std;

torch::Tensor Resnet::INPUT_BOARD(torch::zeros({1, 1, go::N, go::N}));

Resnet::Resnet(float puct, const std::string &model) : Policy(puct)
{
    _module = torch::jit::load(model);
}

void Resnet::get(Position *position, std::vector<Position *> &positions)
{
    updateInputBoard(position);

    vector<torch::jit::IValue> inputs;
    inputs.push_back(INPUT_BOARD);

    auto outputs = _module->forward(inputs);
    auto datalist = outputs.toDoubleListRef();

    for (int i = 0; i <= go::LN; ++i)
    {
        _sortArray[i] = i;
    }

    quickSort(datalist, 0, go::LN, _sortArray);

    position->updateGroup();
    positions.push_back(position->move(go::PASS));
    for (int i = 0; i <= go::LN; ++i)
    {
        if (_sortArray[i] != go::LN)
        {
            int v = go::COORDS[_sortArray[i]];
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

        vector<torch::jit::IValue> inputs;
        inputs.push_back(INPUT_BOARD);
        auto outputs = _module->forward(inputs).toDoubleListRef();

        for (int i = 0; i <= go::LN; ++i)
        {
            _sortArray[i] = i;
        }
        quickSort(outputs, 0, go::LN, _sortArray);

        position->updateGroup();
        Position *pp = nullptr;
        for (int i = 0; i <= go::LN; ++i)
        {
            if (_sortArray[i] != go::LN)
            {
                int v = go::COORDS[_sortArray[i]];
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
    int c = 0;
    int x = 0;
    int y = 0;
    while (c < go::LN)
    {
        int v = go::COORDS[c];

        if (v == position->ko())
        {
            INPUT_BOARD[0][0][y][x] = 2.0;
        }
        else
        {
            INPUT_BOARD[0][0][y][x] = float(position->boardi(v) * position->next());
        }

        x++;
        if (x == go::N)
        {
            y++;
            x = 0;
        }
        c = y * go::N + x;
    }
}

void Resnet::quickSort(const std::vector<double> &arr, int l, int r, int *idx)
{
    int tempi;
    int i, j, x;
    if (l < r)
    {
        i = l;
        j = r;
        x = arr[idx[(l + r) / 2]];
        while (1)
        {
            while (i <= r && arr[idx[i]] < x)
                i++;
            while (j >= 0 && arr[idx[j]] > x)
                j--;
            if (i >= j)
                break;
            else
            {
                tempi = idx[i];
                idx[i] = idx[j];
                idx[j] = tempi;
            }
        }
        quickSort(arr, l, i - 1, idx);
        quickSort(arr, j + 1, r, idx);
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
            int8_t c = INPUT_BOARD[0][0][i][j].template item<float>();
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