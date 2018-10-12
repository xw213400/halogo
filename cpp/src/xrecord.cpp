
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

using namespace std;
using namespace rapidjson;

int main(int argc, char *argv[])
{
    int count = 1;
    string path = "../data/";

    if (argc >= 2)
    {
        path += argv[1];
    }

    if (argc >= 3)
    {
        count += atoi(argv[2]);
    }

    go::init(9);

    MCTSPlayer *playerBlack = new MCTSPlayer(30.f, new RandMove(40));
    MCTSPlayer *playerWhite = new MCTSPlayer(30.f, new RandMove(40));

    int black_win = 0;
    int white_win = 0;

    for (int c = 0; c != count; ++c)
    {
        auto t = chrono::system_clock::to_time_t(chrono::system_clock::now());
        stringstream filename;
        filename << put_time(localtime(&t), "%y%m%d%H%M%S") << ".json";

        cout << "ready: " << c << " in " << count << ", POSPOOL: " << go::POSITION_POOL.size()
             << ", File:" << filename.str() << endl;

        Document positions(kArrayType);
        Document::AllocatorType &allocator = positions.GetAllocator();

        MCTSPlayer *player = playerBlack;

        while (go::POSITION->passCount() < 2)
        {
            bool legal = player->move();
            if (!legal)
            {
                cerr << "Illegal move!" << endl;
                break;
            }

            positions.PushBack(go::POSITION->toJSON(allocator), allocator);

            if (player == playerBlack)
            {
                player = playerWhite;
            }
            else
            {
                player = playerBlack;
            }
        }

        float score = go::POSITION->score() - go::KOMI;
        if (score > 0)
        {
            ++black_win;
        }
        else if (score < 0)
        {
            ++white_win;
        }

        playerBlack->clear();
        playerWhite->clear();
        go::clear();

        StringBuffer buffer;
        PrettyWriter<StringBuffer> writer(buffer);
        positions.Accept(writer);

        ofstream fs(path + filename.str());
        fs << buffer.GetString() << endl;
        fs.close();
    }

    cout << "black win: " << black_win << ", white win: " << white_win
         << ", draw: " << count - black_win - white_win << endl;
}