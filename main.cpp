#include"data.cpp"//�켶�g#include"data.h"�o��LNK2019���~�A�p�󰣿��аѦ�https://www.cnblogs.com/zwj-199306231519/p/12989829.html

using namespace::std;

int main(int argc, char** argv) {
    string path = argv[1];
    Data<float> df(path, true, true, true);
    cout << df.getMatrix() << endl;
    return 0;
}