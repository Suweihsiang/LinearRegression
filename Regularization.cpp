#include "Regularization.h"

//Lasso實作
Lasso::Lasso() {}

Lasso::~Lasso() {}

void Lasso::set_params(unordered_map<string, double> params) {
	if (params.find("iters") != params.end()) {
		iters = params["iters"];
	}
	if (params.find("error") != params.end()) {
		error = params["error"];
	}
	if (params.find("alpha") != params.end()) {
		alpha = params["alpha"];
	}
}

void Lasso::get_params() {
	cout << "iterations = " << iters << ", error = " << error << ", alpha = " << alpha << endl;
}

void Lasso::fit(MatrixXd& x, VectorXd& y) {
	VectorXd intercept_ = VectorXd::Ones(x.rows());
	x.conservativeResize(x.rows(), x.cols() + 1);
	x.block(0, x.cols() - 1, x.rows(), 1) = intercept_;
	//random_device rd;
	//mt19937 gen(rd());
	/*mt19937 gen(1);
	uniform_real_distribution<double> dis(-1.0, 1.0);
	coef = VectorXd::NullaryExpr(x.cols(), [&]() {return dis(gen); });*/
	coef = VectorXd::Zero(x.cols());
	VectorXd errors(x.rows());
	for (int i = 0; i < iters; i++) {
		for (int i = 0; i < x.cols() ; i++) {
			VectorXd x_i = x.col(i);
			double xi_2 = x_i.transpose() * x_i;
			double delta_i = x_i.transpose() * (y - x * coef + coef(i, 0) * x_i);
			if (i < x.cols() - 1) {
				if (delta_i > alpha) {
					coef(i, 0) = (delta_i - alpha) / xi_2;
				}
				else if (delta_i < -alpha) {
					coef(i, 0) = (delta_i + alpha) / xi_2;
				}
				else {
					coef(i, 0) = 0;
				}
			}
			else {
				coef(i, 0) = delta_i / xi_2;
			}
		}
		errors = y - x * coef;
		history.push_back((errors.dot(errors)) / 2);
		if (errors.array().abs().maxCoeff() < error) {
			break;
		}
	}
}

VectorXd Lasso::predict(MatrixXd& x, VectorXd& coef) {
	return x * coef;
}

double Lasso::score(MatrixXd& x, VectorXd& y) {
	if (!coef.array().maxCoeff() && !coef.array().minCoeff()) {
		return 0.0;
	}
	VectorXd y_pred = x * coef;
	VectorXd y_err = y - y_pred;
	VectorXd y_mean = VectorXd::Ones(y.rows()) * y.mean();
	double u = y_err.dot(y_err);
	double v = (y - y_mean).dot(y - y_mean);
	r_2 = (1 - u / v);
	return r_2;
}

VectorXd Lasso::getCoef() const { return coef; }

vector<double> Lasso::gethistory() const { return history; }

//Ridge實作
Ridge::Ridge() {}

Ridge::~Ridge() {}

void Ridge::set_params(unordered_map<string, double> params) {
	if (params.find("alpha") != params.end()) {
		alpha = params["alpha"];
	}
}

void Ridge::get_params() {
	cout <<  "alpha = " << alpha << endl;
}

void Ridge::fit(MatrixXd& x, VectorXd& y) {
	VectorXd intercept_ = VectorXd::Ones(x.rows());
	x.conservativeResize(x.rows(), x.cols() + 1);
	x.block(0, x.cols() - 1, x.rows(), 1) = intercept_;
	MatrixXd alpha_I = alpha * MatrixXd::Identity(x.cols(), x.cols());
	alpha_I(x.cols() - 1, x.cols() - 1) = 0;
	coef = (x.transpose() * x + alpha_I).inverse() * x.transpose() * y;
}