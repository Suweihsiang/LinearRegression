#include "LinearRegression.h"
#include "utils.cuh"
//開始實作函式
LinearRegression::LinearRegression() {}

LinearRegression::~LinearRegression() {}

void LinearRegression::set_params(unordered_map<string,double> params) {
	if (params.find("iters") != params.end()) {
		iters = params["iters"];
	}
	if (params.find("lr") != params.end()) {
		lr = params["lr"];
	}
	if (params.find("error") != params.end()) {
		error = params["error"];
	}
}

void LinearRegression::get_params() {
	cout << "iterations = " << iters << ", learning rate = " << lr << ", error = " << error << endl;
}

void LinearRegression::fit_gd(MatrixXd& x, VectorXd& y, bool CUDA_use,bool fit_intercept) {
	if (fit_intercept) {
		VectorXd intercept_ = VectorXd::Ones(x.rows());
		x.conservativeResize(x.rows(), x.cols() + 1);
		x.block(0, x.cols() - 1, x.rows(), 1) = intercept_;
	}
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> dis(-1.0, 1.0);
	coef = VectorXd::NullaryExpr(x.cols(), [&]() {return dis(gen); });
	cout << coef.transpose() << endl;
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
			coef = coef - 2 * lr * delta / x.rows();
			VectorXd err_hist(1);
			MatrixXd errT = errors.transpose();
			matvecmul_shared(errT, errors, err_hist);
			history.push_back(err_hist(0) / 2);
		}
		else {
			y_pred = x * coef;
			errors = y - y_pred;
			delta = -2 * x.transpose() * errors / x.rows();
			coef = coef - lr * delta;
			history.push_back((errors.dot(errors)) / 2);
		}
	}
}


void LinearRegression::fit_closed_form(MatrixXd& x, VectorXd& y, bool CUDA_use, bool fit_intercept) {
	if (fit_intercept) {
		VectorXd intercept_ = VectorXd::Ones(x.rows());
		x.conservativeResize(x.rows(), x.cols() + 1);
		x.block(0, x.cols() - 1, x.rows(), 1) = intercept_;
	}
	if (CUDA_use) {
		MatrixXd xT = x.transpose();
		MatrixXd xTx(x.cols(),x.cols());
		VectorXd xTy(y.rows());
		coef = VectorXd(x.cols());
		matmul_shared(xT, x, xTx);
		MatrixXd xTx_inv = xTx.inverse();
		matmul_shared(xT, y, xTy);
		matmul_shared(xTx_inv, xTy, coef);
	}
	else {
		coef = (x.transpose() * x).inverse() * x.transpose() * y;
	}
}

VectorXd LinearRegression::predict(MatrixXd& x, VectorXd& coef, bool CUDA_use) {
	VectorXd y_pred(x.rows());
	if (CUDA_use) {
		matmul_shared(x, coef, y_pred);
	}
	else {
		y_pred = x * coef;
	}
	return y_pred;
}

double LinearRegression::score(MatrixXd& x, VectorXd& y) {
	VectorXd y_pred = x * coef;
	VectorXd y_err = y_pred - y;
	VectorXd y_mean = VectorXd::Ones(y.rows()) * y.mean();
	double u = y_err.dot(y_err);
	double v = (y - y_mean).dot(y - y_mean);
	r_2 = (1 - u / v);
	return r_2;
}

VectorXd LinearRegression::getCoef() const { return coef; }

vector<double> LinearRegression::gethistory() const { return history; }
