#include"data.cpp"//原撰寫#include"data.h"發生LNK2019錯誤，如何除錯請參考https://www.cnblogs.com/zwj-199306231519/p/12989829.html

using namespace::std;

int main(int argc, char** argv) {
    string path = argv[1];
    string path2 = argv[2];
    Data<float> df(path, true, true);
    Data<float> df2(path2, true, false);
    df.merge(df2);
    df.print();
    return 0;
}