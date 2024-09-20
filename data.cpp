#include"data.h"

template<typename T>
Data<T>::Data(string path,bool FeatureFirst,bool IndexFirst,bool isThousand) {
	ifstream ifs(path);
	string data;
	bool hasFeature = FeatureFirst;
	hasFeatureColumns = FeatureFirst;
	hasIndexRows = IndexFirst;
	getline(ifs, data);
	if (hasFeature) {
		features = split(data, ',');
		columns = IndexFirst ? features.size() - 1 : features.size();
	}
	mat.conservativeResize(0, columns);
	int r = 0, c = 0;
	while (getline(ifs, data)) {
		mat.conservativeResize(mat.rows() + 1, columns);
		if (isThousand) {
			if (hasIndexRows) {
				indexes.push_back(data.substr(0,data.find(',')));
			}
			vector<string>cols = camma_remove(data);
			for (string col : cols) {
				istringstream iss(col);
				T temp_col;
				iss >> temp_col;
				mat(r,c) = temp_col;
				c++;
			}
		}
		else {
			vector<string> cols = split(data, ',');
			if (hasIndexRows) {
				indexes.push_back(cols[0]);
				cols.erase(cols.begin());
			}
			for (string col : cols) {
				istringstream iss(col);
				T temp_col;
				iss >> temp_col;
				mat(r,c) = temp_col;
				c++;
			}
		}
		r++;
		c = 0;
	}
	rows = mat.rows();
}

template<typename T>
vector<string> Data<T>::split(string s, char dec) {
	istringstream iss(s);
	string subs;
	vector<string>sv;
	while (getline(iss, subs, dec)) {
		sv.push_back(subs);
	}
	return sv;
}

template<typename T>
int Data<T>::getRows() const {
	return rows;
}

template<typename T>
int Data<T>::getColumns() const {
	return columns;
}

template<typename T>
Matrix<T, Dynamic, Dynamic> Data<T>::getMatrix() const {
	return mat;
}

template<typename T>
vector<string> Data<T>::getFeatures() const {
	return features;
}

template<typename T>
vector<string> Data<T>::getIndexs() const {
	return indexes;
}

template<typename T>
vector<string> Data<T>::camma_remove(string data) {
	vector<string>numeric_data;
	size_t begin, end;
	do {
		begin = data.find('"');
		end = data.substr(begin + 1).find('"');
		string nd = data.substr(begin+1, end);
		do {
			nd.erase(nd.find(','),1);
		} while (nd.find(',') != nd.npos);
		numeric_data.push_back(nd);
		data = data.substr(begin + end + 2);
	} while (data.find('"') != data.npos);
	return numeric_data;
}