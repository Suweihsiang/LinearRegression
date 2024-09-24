#include"data.cpp"//原撰寫#include"data.h"發生LNK2019錯誤，如何除錯請參考https://www.cnblogs.com/zwj-199306231519/p/12989829.html
#include<Python.h>
#include"matplotlibcpp.h"
#include<set>

using namespace::std;
namespace plt = matplotlibcpp;

int main(int argc, char** argv) {
    string path = argv[1];
    string path2 = argv[2];
    Data<float> df(path,true, false);
    vector<string> df_idxs = df.getIndexs();
    vector<string>YM;
    for (string idx : df_idxs) {
        YM.push_back(idx.substr(0, 6));
    }
    unordered_map<string, vector<string>>YMap = { { "YM",YM } };
    df.addColumns(YMap);
    df.setIndex("YM");
    set<string> YM_set(YM.begin(), YM.end());
    vector<double> avgs;
    Matrix<float, Dynamic, Dynamic> mat = df.getMatrix();
    vector<float> close(mat.col(0).data(), mat.col(0).data() + df.getRows());
    vector<string> df_idxs2 = df.getIndexs();
    int pos = 0;
    int r = df.getRows();
    for (string ym : YM_set) {
        double sum = 0;
        int count = 0;
        for (int i = pos; i < r; i++) {
            if (df_idxs2[i] == ym) {
                sum += close[i];
                count++;
                pos++;
            }
            else {
                break;
            }
        }
        avgs.push_back(sum / count);
    }
    vector<string>YM_str(YM_set.begin(), YM_set.end());
    vector<string>avg_str;
    for (auto avg : avgs) {
        avg_str.push_back(to_string(avg));
    }
    unordered_map<string, vector<string>>mav = { { "date",YM_str} ,{ "avg",avg_str} };
    Data<float> df_avg(mav,true);
    Data<float> df2(path2, true, false);
    df_avg.merge(df2);
    df_avg.print();

    Matrix<float, Dynamic, Dynamic> mar = df_avg.getMatrix();
    vector<float> x(mar.col(0).data(), mar.col(0).data() + mar.col(0).rows());
    vector<float> y(mar.col(1).data(), mar.col(1).data() + mar.col(1).rows());
    plt::scatter(x, y);
    plt::title("USDTWD vs USUL");
    plt::xlabel("USDTWD");
    plt::ylabel("USUL");
    plt::show();

    return 0;
}