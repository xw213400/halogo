
#include "tensorflow/core/public/session.h"

using namespace std;
using namespace tensorflow;

int main(int argc, char *argv[])
{
    cout << "UIUIUIUI" << endl;
    
    Session *session;
    Status status = NewSession(SessionOptions(), &session);

    cout << "xxxxx" << endl;

    return 0;
}