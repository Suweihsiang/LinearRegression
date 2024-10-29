#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <iostream>
#include<eigen3/Eigen/Dense>
#include<vector>
#include<random>
#include<unordered_map>
#include<string>

using std::vector;
using std::unordered_map;
using std::string;
using std::cout;
using std::endl;
using std::random_device;
using std::mt19937;
using std::uniform_real_distribution;

using Eigen::VectorXd; 
using Eigen::MatrixXd;
using Eigen::Dynamic;
using Eigen::ArrayXd;

class LinearRegression {
public:
	LinearRegression();
	~LinearRegression();
	void set_params(unordered_map<string,double> params);
	void get_params();
	void fit_gd(MatrixXd& x, VectorXd& y, bool CUDA_use, bool fit_intercept);
	void fit_closed_form(MatrixXd& x, VectorXd& y, bool CUDA_use,bool fit_intercept);
	VectorXd predict(MatrixXd& x, VectorXd& coef, bool CUDA_use);
	VectorXd getCoef() const;
	double score(MatrixXd& x, VectorXd& y);
	vector<double> gethistory() const;
protected:
	int iters = 1000;
	double lr = 0.01;
	double error = 0;
	double r_2;
	VectorXd coef;
	vector<double>history;
};
#endif