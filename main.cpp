#include"data.h"
#include"LinearRegression.h"
#include"Regularization.h"
#include<Python.h>
#include"matplotlibcpp.h"
#include<cuda.h>
#include<cuda_runtime.h>
#include "utils.cuh"

namespace plt = matplotlibcpp;

int main(int argc, char** argv) {
    string path = argv[1];
    Data<double> df(path, true, false);
    vector<string> df_idxs = df.getIndexs();
    vector<string>YM;
    for (string idx : df_idxs) {
        YM.push_back(idx.substr(0, 6));
    }
    unordered_map<string, vector<string>>YMap = { {"YM",YM} };
    df.addColumns(YMap);
    df.setIndex("YM");
    df.removeColumn(df.getColumns() - 1);
    df.removeColumns({ "open","high","low","rate" });
    df.renameColumns({ { "close",df.getDataname() } });
    Data<double> df_avg = df.groupby("YM", "mean");
    for (int i = 1; i < argc - 5; i++) {
        Data<double> df(argv[i + 1], true, true);
        df_idxs = df.getIndexs();
        YM.clear();
        for (string idx : df_idxs) {
            YM.push_back(idx.substr(0, 6));
        }
        unordered_map<string, vector<string>>YMap = { {"YM",YM} };
        df.addColumns(YMap);
        df.setIndex("YM");
        df.removeColumn(df.getColumns() - 1);
        df.removeColumns({ "open","high","low" });
        df.renameColumns({ { "close",df.getDataname() } });
        Data<double> df_avg2 = df.groupby("YM", "mean");
        df_avg.merge(df_avg2);
    }
    Data<double> df2(argv[argc - 4], true, false);
    df_idxs = df2.getIndexs();
    YM.clear();
    for (string idx : df_idxs) {
        YM.push_back(idx.substr(0, 6));
    }
    YMap = { {"YM",YM} };
    df2.addColumns(YMap);
    df2.setIndex("YM");
    df2.removeColumn(df2.getColumns() - 1);
    df2.removeColumns({ "MA20","MA60","MA120","MA240" });
    df2.renameColumns({ { "Price",df2.getDataname() } });
    Data<double> df_avg3 = df2.groupby("YM", "mean");
    df_avg3.setDataname(df2.getDataname());
    df_avg.merge(df_avg3);
    Data<double> df3(argv[argc - 3], true, false);
    df_idxs = df3.getIndexs();
    YM.clear();
    for (string idx : df_idxs) {
        YM.push_back(idx.substr(0, 6));
    }
    YMap = { {"YM",YM} };
    df3.addColumns(YMap);
    df3.setIndex("YM");
    df3.removeColumn(df3.getColumns() - 1);
    df3.removeColumns({ "MA20","MA60","MA120" });
    df3.renameColumns({ { "Price",df3.getDataname() } });
    Data<double> df_avg4 = df3.groupby("YM", "mean");
    df_avg4.setDataname(df3.getDataname());
    df_avg.merge(df_avg4);
    Data<double> df4(argv[argc - 2], true, false);
    df_idxs = df4.getIndexs();
    YM.clear();
    for (string idx : df_idxs) {
        YM.push_back(idx.substr(0, 6));
    }
    YMap = { {"YM",YM} };
    df4.addColumns(YMap);
    df4.setIndex("YM");
    df4.removeColumn(df4.getColumns() - 1);
    df4.removeColumns({ "open","high","low","rate" });
    df4.renameColumns({ { "close","US10Y"} });
    Data<double> df_avg5 = df4.groupby("YM", "mean");
    df_avg5.setDataname(df4.getDataname());
    df_avg.merge(df_avg5);
    Data<double> df5(argv[argc - 1], true, false);
    df_avg.merge(df5);

    Matrix<double, Dynamic, Dynamic> mar = df_avg.getMatrix();

    MatrixXd x = mar.block(0, 0, mar.rows(), 8);
    VectorXd y = mar.block(0, 8, mar.rows(), 1);

    for (int i = 0; i < x.cols(); i++) {
        double xstd = (x.col(i) - x.col(i).mean() * VectorXd::Ones(x.rows())).transpose() * (x.col(i) - x.col(i).mean() * VectorXd::Ones(x.rows()));
        x.col(i) = (x.col(i) - x.col(i).mean() * VectorXd::Ones(x.rows())) / std::sqrt(xstd / x.rows());
    }
    double ystd = (y - y.mean() * VectorXd::Ones(y.rows())).transpose() * (y - y.mean() * VectorXd::Ones(y.rows()));
    y = (y - y.mean() * VectorXd::Ones(y.rows())) / std::sqrt(ystd / y.rows());


    Lasso_LARS reg;
    reg.set_params({ {"iters",10000},{"alpha",0} });
    reg.get_params();
    reg.fit(x, y, "bic", true);
    vector<double> bics = reg.get_criterions();
    Lasso_LARS reg1;
    reg1.set_params({ {"iters",10000},{"alpha",0} });
    reg1.get_params();
    reg1.fit(x, y, "aic", true);
    vector<double> aics = reg1.get_criterions();
    plt::plot(aics, { {"label", "AIC"} });
    plt::plot(bics, { {"label", "BIC"} });
    plt::xlabel("sequence");
    plt::ylabel("criterion");
    plt::legend();
    plt::title("AIC vs BIC");
    plt::show();

    return 0;
}
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
/*
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
    vector<double>log_as, w1, w2, w3, w4,aic_v,bic_v;
    for (double a : as) {
        cout << "----------------------------------------" << endl;
        reg.set_params({ {"iters",10000},{"error",0.01},{"alpha",a} });
        reg.get_params();
        MatrixXd x = mar.block(0, 0, mar.rows(), 4);
        VectorXd y = mar.block(0, 6, mar.rows(), 1);
        reg.fit(x, y);
        VectorXd coef = reg.getCoef();
        double aic = reg.calc_IC(x, y);
        double bic = reg.calc_IC(x, y,"bic");
        cout << "R2 = " << reg.score(x, y) << endl;
        cout << "AIC = " << aic << endl;
        cout << "BIC = " << bic << endl;
        cout << "coef = " << coef.transpose() << endl;
        log_as.push_back(log10(a));
        w1.push_back(coef(0));
        w2.push_back(coef(1));
        w3.push_back(coef(2));
        w4.push_back(coef(3));
        aic_v.push_back(aic);
        bic_v.push_back(bic);
    }
    /*plt::plot(log_as, w1);
    plt::plot(log_as, w2);
    plt::plot(log_as, w3);
    plt::plot(log_as, w4);
    plt::xlabel("log alpha");
    plt::ylabel("weight");
    plt::title("weight vs log(alpha)");*//*
    plt::plot(log_as, aic_v,{{"label", "AIC"}});
    plt::plot(log_as, bic_v, { {"label", "BIC"} });
    plt::xlabel("log alpha");
    plt::ylabel("criterion");
    plt::legend();
    plt::title("criterion vs log(alpha)");
    plt::show();

    return 0;
}
*/