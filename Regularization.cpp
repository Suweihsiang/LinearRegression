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
	if (params.find("alpha") != params.end()) {
		alpha = params["alpha"];
	}
}

void Lasso_LARS::get_params() {
	cout << "iterations = " << iters << ", error = " << error << ", alpha = " << alpha << endl;
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
	vector<double>alpha_path;
	VectorXd sign_active = VectorXd::Zero(x.cols());
	bool changed_gamma = false;
	coef_path.push_back(coef);
	int next_j = -1;
	int it = 0;
	while (it < max_feature) {
		VectorXd y_pred = x * coef;
		VectorXd corr = x.transpose() * (y - y_pred);
		double C = corr.array().abs().maxCoeff();

		double C_alpha = corr.maxCoeff();
		double alpha_ = C_alpha / m;
		alpha_path.push_back(alpha_);

		if (alpha_ < alpha) {
			if (alpha_path.size() > 1) {
				double ss = (*(alpha_path.end() - 2) - alpha) / (*(alpha_path.end() - 2) - alpha_);
				VectorXd prev_coef = *(coef_path.end() - 2);
				coef = prev_coef + ss * (coef - prev_coef);
				coef_path.back() = coef;
			}
			alpha_path.back() = alpha;
			break;
		}

		if (!changed_gamma) {
			double Cabs_max = 0.0;
			for (int i = 0; i < in_active.size(); i++) {
				int idx = in_active[i];
				double Cabs_idx = abs(corr[idx]);
				if (Cabs_idx > Cabs_max || (i == 0 && Cabs_idx >= Cabs_max)) {
					Cabs_max = Cabs_idx;
					next_j = idx;
				}
			}
			active.push_back(next_j);
			sign_active(active.size() - 1) = corr[next_j] / abs(corr[next_j]);
			if (in_active.size() > 0) {
				in_active.erase(find(in_active.begin(), in_active.end(), next_j));
			}
		}
		MatrixXd Xa(x.rows(), active.size());
		int i = 0;
		while (i < active.size()) {
			int idx = active[i];
			Xa.col(i) = x.col(idx);
			i++;
		}
		MatrixXd Ga = Xa.transpose() * Xa;
		VectorXd sign_act = sign_active.head(active.size());
		VectorXd Ga_inv = Ga.inverse() * sign_act;
		double Aa = 1 / sqrt((Ga_inv.array() * sign_act.array()).sum());
		VectorXd Wa = Aa * Ga_inv;
		VectorXd ua = Xa * Wa;
		VectorXd a = x.transpose() * ua;
		double gamma = 0.0;
		for (int j = 0; j < n; j++) {
			if (find(active.begin(), active.end(), j) != active.end()) { continue; }
			double v0 = (C - corr(j)) / (Aa - a(j));
			double v1 = (C + corr(j)) / (Aa + a(j));
			if (v0 > 0 && (gamma == 0 || v0 < gamma)) {
				gamma = v0;
			}
			if (v1 > 0 && (gamma == 0 || v1 < gamma)) {
				gamma = v1;
			}
		}
		if (gamma == 0.0 || C / Aa < gamma) {
			gamma = C / Aa;
		}
		double gamma_sc = 0.0;
		int sc_j;
		int sign_i;
		for (int i = 0; i < active.size(); i++) {
			int idx = active[i];
			double d = Wa[i];
			double c = coef[idx];
			double g = -c / d;

			if (g > 0 && (gamma_sc == 0.0 || g < gamma_sc)) {
				gamma_sc = g;
				sc_j = idx;
				sign_i = i;
			}
		}
		changed_gamma = false;
		if (gamma_sc > 0 && gamma_sc < gamma) {
			next_j = sc_j;
			gamma = gamma_sc;
			sign_active[sign_i] = -sign_active[sign_i];
			changed_gamma = true;
		}
		for (int j = 0; j < active.size(); j++) {
			int idx = active[j];
			coef(idx) += Wa(j) * gamma;
		}
		if (changed_gamma) {
			active.erase(find(active.begin(), active.end(), next_j));
			in_active.push_back(next_j);
			if (sign_i != sign_active.rows() - 1) {
				sign_active.block(sign_i, 0, sign_active.rows() - 1 - sign_i, 1) = sign_active.block(sign_i + 1, 0, sign_active.rows() - 1 - sign_i, 1).eval();
			}
			sign_active[sign_active.rows() - 1] = 0;
		}
		coef_path.push_back(coef);
		it = active.size();
	}
	if (alpha_path.size() != coef_path.size()) {
		alpha_path.push_back((x.transpose()* (y - x * coef)).maxCoeff() / m);
	}

	for (int i = 0; i < alpha_path.size(); i++) {
		cout << alpha_path[i] << " ";
	}
	cout << endl;
	for (int i = 0; i < coef_path.size(); i++) {
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