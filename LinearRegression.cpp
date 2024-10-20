#include "LinearRegression.h"
#include "utils.cuh"
//開始實作函式
LinearRegression::LinearRegression() {}

LinearRegression::LinearRegression(int _iter, double _lr, double _error) :iters(_iter), lr(_lr), error(_error) {}

LinearRegression::~LinearRegression() {}

void LinearRegression::fit(MatrixXd& x, VectorXd& y, bool CUDA_use,bool fit_intercept) {
	if (fit_intercept) {
		VectorXd intercept_ = VectorXd::Ones(x.rows());
		x.conservativeResize(x.rows(), x.cols() + 1);
		x.block(0, x.cols() - 1, x.rows(), 1) = intercept_;
	}
	coef = VectorXd::Random(x.cols());
	VectorXd y_pred(x.rows());
	VectorXd errors(x.rows());
	VectorXd delta(x.cols());
	for (int i = 0; i < iters; i++) {
		if (errors.array().abs().maxCoeff() < error) {
			break;
		}
		if (CUDA_use) {
			matvecmul_shared(x, coef, y_pred);
			errors = y - y_pred;
			MatrixXd xT = x.transpose();
			matvecmul_shared(xT, errors, delta);
			coef = coef + lr * delta;
			VectorXd err_hist(1);
			MatrixXd errT = errors.transpose();
			matvecmul_shared(errT, errors, err_hist);
			history.push_back(err_hist(0) / 2);
		}
		else {
			y_pred = x * coef;
			errors = y - y_pred;
			delta = x.transpose() * errors;
			coef = coef + lr * delta;
			history.push_back((errors.dot(errors)) / 2);
		}
	}
}

VectorXd LinearRegression::predict(MatrixXd& x, VectorXd& coef, bool CUDA_use) {
	VectorXd y_pred(x.rows());
	if (CUDA_use) {
		matvecmul_shared(x, coef, y_pred);
	}
	else {
		y_pred = x * coef;
	}
	return y_pred;
}

double LinearRegression::score(MatrixXd& x, VectorXd& y) {
	VectorXd y_pred = x * coef;
	VectorXd y_err = y_pred - y;
	double u = y_err.dot(y_err);
	double v = y.dot(y);
	return (1 - u / v);
}

VectorXd LinearRegression::getCoef() const { return coef; }

vector<double> LinearRegression::gethistory() const { return history; }