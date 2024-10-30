#ifndef REGULARIZATION_H
#define REGULARIZATION_H

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
	vector<double> gethistory() const;
protected:
	double alpha = 100;
	double r_2;
	VectorXd coef;
	vector<double>history;
private:
	int iters = 1000;
	double error = 0;
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