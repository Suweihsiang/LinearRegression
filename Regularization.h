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

class Lasso {
public:
	Lasso();
	~Lasso();
	void set_params(unordered_map<string, double> params);
	void get_params();
	void fit(MatrixXd& x, VectorXd& y);
	VectorXd predict(MatrixXd& x, VectorXd& coef);
	VectorXd getCoef() const;
	double score(MatrixXd& x, VectorXd& y);
	double calc_IC(MatrixXd& x, VectorXd& y,string criterion = "aic");
	vector<double> gethistory() const;
	void save_result(string path, string mode = "app");
protected:
	double alpha = 100;
	double r_2;
	double IC;
	VectorXd coef;
	vector<double>history;
	double calc_noise_var(MatrixXd& x, VectorXd& y);
	double get_degree_of_freedom(VectorXd& coef);
private:
	int iters = 1000;
	double error = 0;
};

class Lasso_LARS {
public:
	Lasso_LARS();
	~Lasso_LARS();
	void set_params(unordered_map<string, double> params);
	void get_params();
	void fit(MatrixXd& x, VectorXd& y,string criterion,bool fit_intercept);
	double calc_IC(MatrixXd& x, VectorXd& y, VectorXd& coef, string criterion, bool fit_intercept, double noise_var);
	vector<VectorXd> get_coef_path() const;
	vector<double> get_alpha_path() const;
	vector<double> get_criterions() const;
	void save_result(string path);
private:
	int iters = 10000;
	double alpha_min = 0;
	vector<VectorXd>coef_path;
	vector<double>alpha_path;
	vector<double>criterions;
	double calc_noise_var(MatrixXd& x, VectorXd& y, bool fit_intercept);
	double get_degree_of_freedom(VectorXd& coef,bool fit_intercept);
	double noise_variance = -1.0;
	double alpha;
	double IC;
	VectorXd best_coef;
};

class Ridge : public Lasso{
public:
	Ridge();
	~Ridge();
	void set_params(unordered_map<string, double> params);
	void get_params();
	void fit(MatrixXd& x, VectorXd& y);
};
#endif