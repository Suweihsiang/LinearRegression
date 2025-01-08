#ifndef REGULARIZATION_H
#define REGULARIZATION_H

#define _USE_MATH_DEFINES

#include <iostream>
#include<fstream>
#include<eigen3/Eigen/Dense>
#include<eigen3/unsupported/Eigen/matrixfunctions>
#include<vector>
#include<algorithm>
#include<random>
#include<unordered_map>
#include<string>
#include<cmath>
#include<numeric>

using std::vector;
using std::unordered_map;
using std::string;
using std::ofstream;
using std::ios;
using std::cout;
using std::endl;
using std::sort;
using std::min;
using std::random_device;
using std::mt19937;
using std::uniform_real_distribution;

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Dynamic;
using Eigen::ArrayXd;
using Eigen::Index;

class Lasso {														   //coordinate descent for lasso
public:
	Lasso();														   //constructor
	~Lasso();														   //destructor
	void set_params(unordered_map<string, double> params);			   //set parameters
	void get_params();												   //show parameters
	void fit(MatrixXd& x, VectorXd& y);
	VectorXd predict(MatrixXd& x, VectorXd& coef);
	VectorXd getCoef() const;
	double score(MatrixXd& x, VectorXd& y);							   //calculate R2
	double calc_IC(MatrixXd& x, VectorXd& y,string criterion = "aic"); //calculate AIC,BIC
	vector<double> gethistory() const;								   //errors of each epoch
	void save_result(string path, string mode = "app");				   //save result
protected:
	double alpha = 100;												   //L1 penalty(Lasso),L2 penalty(Ridge)
	double r_2;
	double IC;														   //AIC or BIC
	VectorXd coef;
	vector<double>history;
	double calc_noise_var(MatrixXd& x, VectorXd& y);				   //calculate for noise variance in order to calculate AIC,BIC
	double get_degree_of_freedom(VectorXd& coef);					   //get degree of freedom inorder to calculate AIC,BIC
private:
	int iters = 1000;
	double error = 0;
};

class Lasso_LARS {														//least angle regression algorithm
public:
	Lasso_LARS();//constructor
	~Lasso_LARS();//destructor
	void set_params(unordered_map<string, double> params);				//set parameters
	void get_params();													//show parameters
	void fit(MatrixXd& x, VectorXd& y,string criterion,bool fit_intercept);
	double calc_IC(MatrixXd& x, VectorXd& y, VectorXd& coef, string criterion, bool fit_intercept, double noise_var);//calculate AIC,BIC
	vector<VectorXd> get_coef_path() const;								//get corresponding coefficients of each lasso path
	vector<double> get_alpha_path() const;								//get corresponding L1 penalty of each lasso path
	vector<double> get_criterions() const;								//get corresponding AIC,BIC of each lasso path
	void save_result(string path);										//save result as csv file
private:
	int iters = 10000;
	double alpha_min = 0;
	vector<VectorXd>coef_path;											//corresponding coefficients of each lasso path
	vector<double>alpha_path;											//corresponding L1 penalty of each lasso path
	vector<double>criterions;											//corresponding AIC,BIC of each lasso path
	double calc_noise_var(MatrixXd& x, VectorXd& y, bool fit_intercept);//calculate for noise variance in order to calculate AIC,BIC
	double get_degree_of_freedom(VectorXd& coef,bool fit_intercept);	//get degree of freedom inorder to calculate AIC,BIC
	double noise_variance = -1.0;
	double alpha;														//L1 penalty
	double IC;															//AIC or BIC
	VectorXd best_coef;
};

class Ridge : public Lasso{									//inherit some members and function from Lasso
public:
	Ridge();												//constructor
	~Ridge();												//destructor
	void set_params(unordered_map<string, double> params);  //set parameters
	void get_params();										//show parameters
	void fit(MatrixXd& x, VectorXd& y);
};
#endif