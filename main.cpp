#include"data.cpp"//原撰寫#include"data.h"發生LNK2019錯誤，如何除錯請參考https://www.cnblogs.com/zwj-199306231519/p/12989829.html
#include<Python.h>
#include"matplotlibcpp.h"

using namespace::std;
namespace plt = matplotlibcpp;

int main(int argc, char** argv) {
    string path = argv[1];
    string path2 = argv[2];
    Data<float> df(path, true, true);
    Data<float> df2(path2, true, false);
    df.merge(df2);
    df.print();

    Matrix<float, Dynamic, Dynamic> mat = df.getMatrix();
    std::vector<float> x(mat.col(0).data(), mat.col(0).data() + mat.col(0).rows());
    std::vector<float> y(mat.col(4).data(), mat.col(4).data() + mat.col(4).rows());
    plt::scatter(x,y);
    plt::title("NASDAQ vs MSCI");
    plt::xlabel("NASDAQ");
    plt::ylabel(" MSCI");
    plt::show();

    return 0;
}