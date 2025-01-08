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
	LinearRegression();																 //constructor
	~LinearRegression();															 //destructor
	void set_params(unordered_map<string,double> params);							 //set parameters
	void get_params();																 //show the parameters
	void fit_gd(MatrixXd& x, VectorXd& y, bool CUDA_use, bool fit_intercept);	     //fit linear regression by gradient descent 
	void fit_closed_form(MatrixXd& x, VectorXd& y, bool CUDA_use,bool fit_intercept);//fit linear regression by closed form
	VectorXd predict(MatrixXd& x, VectorXd& coef, bool CUDA_use);					 //predict
	VectorXd getCoef() const;														 //show the coeficient
	double score(MatrixXd& x, VectorXd& y);											 //get R2
	vector<double> gethistory() const;												 //errors of each epoch by gradient descent
protected:
	int iters = 1000;															     //gradient descent iterations
	double lr = 0.01;																 //gradient descent learning rate
	double error = 0;
	double r_2;
	VectorXd coef;
	vector<double>history;
};
#endif