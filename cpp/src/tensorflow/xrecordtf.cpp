
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include "go.h"
#include "MCTSPlayer.h"
#include "RandMove.h"
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "tf/Resnet.h"
#include "DTPlayer.h"

using namespace std;
using namespace rapidjson;

int main(int argc, char *argv[])
{
    int count = 1;
    string path = "../../data/";

    if (argc >= 2)
    {
        path += argv[1];
        path += "/";
    }

    if (argc >= 3)
    {
        count = atoi(argv[2]);
    }

    go::init();

    NNParam paramA;
    NNParam paramB;

    paramA.pdfile = "../../data/goai.pb";
    paramA.branches = 20;
    paramA.simstep = 8;
    paramA.simrand = 0.5;

    // paramB.pdfile = "../../data/goai.pb";
    // paramB.branches = 20;
    // paramB.simstep = 0;
    // paramB.simrand = 0.5;

    MCTSPlayer::POOL.resize(50000);
    DTPlayer::POOL.resize(50000);

    MCTSPlayer *playerB = new MCTSPlayer(new Resnet(&paramA), 900);
    // MCTSPlayer *playerB = new MCTSPlayer(1000, new RandMove(0.5f));
    // MCTSPlayer *playerB = new MCTSPlayer(3000, new RandMove(0.5f));
    // MCTSPlayer *playerB = new MCTSPlayer(new Resnet(&paramB), 1050);
    DTPlayer *playerA = new DTPlayer("../../data/goai.pb", 12);

    int a_win = 0;
    int b_win = 0;
    bool swap = true;

    for (int c = 0; c != count; ++c)
    {
        auto t = chrono::system_clock::to_time_t(chrono::system_clock::now());
        stringstream filename;
        filename << put_time(localtime(&t), "%y%m%d%H%M%S") << ".json";

        swap = !swap;

        cout << "ready: " << c + 1 << " in " << count
             << ", Swap:" << swap
             << ", PP:" << go::POSITION_POOL.size()
             << ", GP:" << Group::POOL.size()
             << ", MP:" << MCTSPlayer::POOL.size()
             << ", File:" << filename.str() << endl;

        Document positions(kArrayType);
        Document::AllocatorType &allocator = positions.GetAllocator();

        Player *player = nullptr;
        if (swap)
        {
            player = playerB;
        }
        else
        {
            player = playerA;
        }

        while (go::POSITION->passCount() < 2)
        {
            bool legal = player->move();
            if (!legal)
            {
                cerr << "Illegal move!" << endl;
                break;
            }

            positions.PushBack(go::POSITION->toJSON(allocator), allocator);

            if (player == playerA)
            {
                player = playerB;
            }
            else
            {
                player = playerA;
            }
        }

        go::POSITION->debug();
        cout << "Score: " << go::POSITION->score() << endl;

        float score = go::POSITION->score() - go::KOMI;
        if (score > 0)
        {
            swap ? ++b_win : ++a_win;
        }
        else if (score < 0)
        {
            swap ? ++a_win : ++b_win;
        }

        cout << "A win: " << a_win << ", B win: " << b_win << endl;

        playerA->clear();
        playerB->clear();
        go::clear();

        StringBuffer buffer;
        Writer<StringBuffer> writer(buffer);
        positions.Accept(writer);

        string fullpath = path + filename.str();
        ofstream fs(fullpath);
        if (fs)
        {
            fs << buffer.GetString() << endl;
            fs.close();
        }
        else
        {
            cerr << "create file '" << fullpath << "' failed" << endl;
        }
    }

    cout << "A win: " << a_win << ", B win: " << b_win
         << ", Draw: " << count - a_win - b_win << endl;
}