#include "Regularization.h"
#include"LinearRegression.h"

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

double Lasso::calc_IC(MatrixXd& x, VectorXd& y, string criterion) {
	double n = x.cols();
	double m = x.rows();
	VectorXd y_pred = predict(x, coef);
	double d = get_degree_of_freedom(coef);
	double noise_var = calc_noise_var(x, y);
	double sse = (y - y_pred).transpose() * (y - y_pred); 
	double factor = (criterion == "aic") ? 2 : log(m);
	IC = factor * d + m * log(2 * M_PI * noise_var) + sse / noise_var;
	return IC;
}

VectorXd Lasso::getCoef() const { return coef; }

vector<double> Lasso::gethistory() const { return history; }

double Lasso::calc_noise_var(MatrixXd& x, VectorXd& y) {
	LinearRegression lr;
	lr.fit_closed_form(x, y, false, false);
	VectorXd coef = lr.getCoef();
	VectorXd y_pred = lr.predict(x, coef, false);
	double error_2 = (y - y_pred).transpose() * (y - y_pred);
	return error_2 / (x.rows() - x.cols());
}

double Lasso::get_degree_of_freedom(VectorXd& coef) {
	double d = coef.rows() - 1;
	for (int i = 0; i < coef.rows() - 1; i++) {
		if (coef(i, 0) == 0) {
			d--;
		}
	}
	return d;
}

//Lasso_LARS實作
Lasso_LARS::Lasso_LARS() {}

Lasso_LARS::~Lasso_LARS(){}

void Lasso_LARS::set_params(unordered_map<string, double> params) {
	if (params.find("iters") != params.end()) {
		iters = params["iters"];
	}
	if (params.find("error") != params.end()) {
		error = params["error"];
	}
}

void Lasso_LARS::get_params() {
	cout << "iterations = " << iters << ", error = " << error << endl;
}

void Lasso_LARS::fit(MatrixXd& x, VectorXd& y, string criterion) {
	int m = x.rows();
	int n = x.cols();
	int max_feature = min(n, iters);
	VectorXd coef = VectorXd::Zero(x.cols());
	vector<int> active;
	vector<int>in_active(x.cols());
	generate(in_active.begin(), in_active.end(), [] {static int in_num = 0; return in_num++; });
	vector<VectorXd>coef_path;
	vector<double>alpha_path(max_feature + 1);
	bool changed_gamma = false;
	coef_path.push_back(coef);
	int next_j = -1;
	int it = 0;
	while (it < max_feature) {
		VectorXd y_pred = x * coef;
		VectorXd corr = x.transpose() * (y - y_pred);
		double C = corr.array().abs().maxCoeff();
		if (!changed_gamma) {
			double Cabs_max = 0.0;
			for (int i = 0; i < in_active.size(); i++) {
				int idx = in_active[i];
				double Cabs_idx = abs(corr[idx]);
				if (Cabs_idx >= Cabs_max) {
					Cabs_max = Cabs_idx;
					next_j = idx;
				}
			}
			active.push_back(next_j);
			if (in_active.size() > 0) {
				in_active.erase(find(in_active.begin(), in_active.end(), next_j));
			}
		}
		MatrixXd Xa(x.rows(), active.size());
		VectorXd sign_active(active.size());
		int i = 0;
		while (i < active.size()) {
			int idx = active[i];
			sign_active[i] = corr(idx) / abs(corr(idx));
			Xa.col(i) = x.col(idx);
			Xa.col(i) *= sign_active(i);
			i++;
		}
		MatrixXd Ga = Xa.transpose() * Xa;
		MatrixXd Ga_inv = Ga.inverse();
		VectorXd Ia = VectorXd::Ones(active.size());
		double Aa = 1 / sqrt(Ga_inv.sum());
		VectorXd Wa = Aa * Ga_inv * Ia;
		VectorXd ua = Xa * Wa;
		VectorXd a = x.transpose() * ua;
		double gamma = 0.0;
		if (it < n - 1) {
			for (int j = 0; j < n; j++) {
				if (find(active.begin(), active.end(), j) != active.end()) {continue;}
				double v0 = (C - corr(j)) / (Aa - a(j));
				double v1 = (C + corr(j)) / (Aa + a(j));
				if (v0 > 0 && (gamma == 0 || v0 < gamma)) {
					gamma = v0;
				}
				if (v1 > 0 && (gamma == 0 || v1 < gamma)) {
					gamma = v1;
				}
			}
		}
		else {
			gamma = C / Aa;
		}
		double gamma_sc = 0.0;
		vector<int> sc_idx;
		for (int i = 0; i < active.size(); i++) {
			int idx = active[i];
			double d = sign_active[i] * Wa[i];
			double c = coef[idx];
			double g = -c / d;
			
			if ( g > 0 && (gamma_sc == 0.0 || g <= gamma_sc)) {
				if (g < gamma_sc) {
					sc_idx = vector<int>();
				}
				gamma_sc = g;
				sc_idx.push_back(idx);
			}
		}
		changed_gamma = false;
		if (gamma_sc > 0 && gamma_sc < gamma) {
			gamma = gamma_sc;
			changed_gamma = true;
		}
		for (int j = 0; j < active.size(); j++) {
			int idx = active[j];
			coef(idx) += Wa(j) * gamma * sign_active[j];
		}
		if (changed_gamma) {
			for (const int i : sc_idx) {
				active.erase(find(active.begin(),active.end(),i));
				in_active.push_back(i);
			}
		}
		coef_path.push_back(coef);
		it = active.size();
	}
	for(int i = 0; i < coef_path.size(); i++) {
		cout << coef_path[i].transpose() << endl;
	}
}

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