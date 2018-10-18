
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

bool readFile(const string &fileName, string &out)
{
    ifstream ifs(fileName.c_str(), ifstream::binary | ifstream::ate);

    if (!ifs.is_open())
    {
        ifs.close();
        return false;
    }

    int nLen = (int)ifs.tellg();

    if (nLen == 0)
    {
        ifs.close();
        return false;
    }

    ifs.seekg(0, ios::beg);

    vector<char> bytes(nLen);
    ifs.read(&bytes[0], nLen);

    ifs.close();

    out.assign(&bytes[0], nLen);

    return true;
}

int main(int argc, char *argv[])
{
    int step = 1;
    string filepath = "../../data/";

    if (argc < 3)
    {
        return 0;
    }

    if (argc >= 2)
    {
        filepath += argv[1];
        filepath += ".json";
    }

    if (argc >= 3)
    {
        step = atoi(argv[2]);
    }

    string record;
    bool bSuccess = readFile(filepath, record);

    if (!bSuccess)
    {
        cerr << "Read file '" << filepath << "' failed!" << endl;
        return 0;
    }

    go::init();
    MCTSPlayer *player = new MCTSPlayer(5000.f, new RandMove(80));

    Document doc;
    doc.Parse(record.c_str());

    for (int i = 0; i != step; ++i)
    {
        int vertex = doc.GetArray()[i]["vertex"].GetInt();
        go::POSITION = go::POSITION->move(vertex);
        go::POSITION->updateGroup();
    }

    go::POSITION->debugGroup();

    cout << endl;

    while (go::POSITION->passCount() < 2)
    {
        go::POSITION->debug();
        cout << "Next:" << to_string(go::POSITION->next()) << endl;

        bool legal = player->move();
        if (!legal)
        {
            cerr << "Illegal move!" << endl;
        }
    }

    return 0;
}