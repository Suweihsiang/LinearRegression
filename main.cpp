#include"data.cpp"//原撰寫#include"data.h"發生LNK2019錯誤，如何除錯請參考https://www.cnblogs.com/zwj-199306231519/p/12989829.html
#include<Python.h>
#include"matplotlibcpp.h"
#include<set>

using namespace::std;
namespace plt = matplotlibcpp;

int main(int argc, char** argv) {
    string path = argv[1];
    string path2 = argv[2];
    Data<float> df(path, true, false);
    vector<string> df_idxs = df.getIndexs();
    vector<string>YM;
    for (string idx : df_idxs) {
        YM.push_back(idx.substr(0, 6));
    }
    unordered_map<string, vector<string>>YMap = { {"YM",YM} };
    df.addColumns(YMap);
    df.setIndex("YM");
    Data<float> df_avg = df.groupby("YM", "mean");
    Data<float> df2(path2, true, false);
    df_avg.merge(df2);
    df_avg.print();

    Matrix<float, Dynamic, Dynamic> mar = df_avg.getMatrix();
    vector<float> x(mar.col(0).data(), mar.col(0).data() + mar.rows());
    vector<float> y(mar.col(6).data(), mar.col(6).data() + mar.rows());
    plt::scatter(x, y);
    plt::title("USDTWD vs USUL");
    plt::xlabel("USDTWD");
    plt::ylabel("USUL");
    plt::show();

    return 0;
}