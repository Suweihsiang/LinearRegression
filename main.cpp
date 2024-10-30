#include"data.h"
#include"LinearRegression.h"
#include"Regularization.h"
#include<Python.h>
#include"matplotlibcpp.h"
#include<cuda.h>
#include<cuda_runtime.h>
#include "utils.cuh"

namespace plt = matplotlibcpp;
/*
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
}*/

int main(int argc, char** argv) {
    string path = argv[1];
    string path2 = argv[2];
    Data<double> df(path, true, false);
    vector<string> df_idxs = df.getIndexs();
    vector<string>YM;
    for (string idx : df_idxs) {
        YM.push_back(idx.substr(0, 6));
    }
    unordered_map<string, vector<string>>YMap = { {"YM",YM} };
    df.addColumns(YMap);
    df.setIndex("YM");
    Data<double> df_avg = df.groupby("YM", "mean");
    Data<double> df2(path2, true, false);
    df_avg.merge(df2);
    //df_avg.print();

    Matrix<double, Dynamic, Dynamic> mar = df_avg.getMatrix();

    //LinearRegression lr;
    Lasso reg;
    //Ridge reg;
    vector<double>as = { 0,1,5,10,50,100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000 };
    vector<double>log_as, w1, w2, w3, w4;
    for (double a : as) {
        cout << "----------------------------------------" << endl;
        reg.set_params({ {"iters",10000},{"error",0.01},{"alpha",a} });
        reg.get_params();
        MatrixXd x = mar.block(0, 0, mar.rows(), 4);
        VectorXd y = mar.block(0, 6, mar.rows(), 1);
        reg.fit(x, y);
        VectorXd coef = reg.getCoef();
        cout << "R2 = " << reg.score(x, y) << endl;
        cout << "AIC = " << reg.calc_AIC(x, y) << endl;
        cout << "BIC = " << reg.calc_BIC(x, y) << endl;
        cout << "coef = " << coef.transpose() << endl;
        log_as.push_back(log10(a));
        w1.push_back(coef(0));
        w2.push_back(coef(1));
        w3.push_back(coef(2));
        w4.push_back(coef(3));
    }
    plt::plot(log_as, w1);
    plt::plot(log_as, w2);
    plt::plot(log_as, w3);
    plt::plot(log_as, w4);
    plt::xlabel("log alpha");
    plt::ylabel("weight");
    plt::title("weight vs log(alpha)");
    plt::show();

    return 0;
}