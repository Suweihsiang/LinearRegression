#include"data.cpp"//原撰寫#include"data.h"發生LNK2019錯誤，如何除錯請參考https://www.cnblogs.com/zwj-199306231519/p/12989829.html

using namespace::std;

int main(int argc, char** argv) {
    string path = argv[1];
    string path2 = argv[2];
    Data<float> df(path, true, true, true);
    //cout << df.getMatrix() << endl;
    Data<float> df2(path2, true, true, false);
    //cout << df2.getMatrix() << endl;
    df.merge(df2);
    vector<string> f1 = df.getFeatures();
    //vector<string> i1 = df.getIndexs();
    for (string f : f1) {
        cout << f << "\t";
    }
    cout << endl;
    /*for (string i : i1) {
        cout << i << "\t";
    }
    cout << endl;*/
    cout << df.getMatrix() << endl;
    return 0;
}