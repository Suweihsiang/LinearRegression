#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <iostream>
#include<eigen3/Eigen/Dense>
#include<vector>

using std::vector;
using std::cout;
using std::endl;

using Eigen::VectorXd; 
using Eigen::MatrixXd;
using Eigen::Dynamic;

class LinearRegression {
public:
	LinearRegression();
	LinearRegression(int _iter, double _lr,double _error);
	~LinearRegression();
	void fit(MatrixXd& x, VectorXd& y, bool CUDA_use,bool fit_intercept);
	VectorXd predict(MatrixXd& x, VectorXd& coef, bool CUDA_use);
	VectorXd getCoef() const;
	double score(MatrixXd& x, VectorXd& y);
	vector<double> gethistory() const;
private:
	int iters = 1000;
	double lr = 0.01;
	double error;
	double r_2;
	VectorXd coef;
	vector<double>history;
};

#endif