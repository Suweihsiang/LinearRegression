#include"data.h"

template<typename T>
Data<T>::Data(string path,bool FeatureFirst,bool IndexFirst,bool isThousand) {
	ifstream ifs(path);
	string data;
	bool hasFeature = FeatureFirst;
	hasFeatureColumns = FeatureFirst;
	hasIndexRows = IndexFirst;
	while (getline(ifs, data)) {
		if (hasFeature) {
			features = split(data, ',');
			columns = IndexFirst ? features.size()-1:features.size();
			hasFeature = false;
			continue;
		}
		if (isThousand) {
			if (hasIndexRows) {
				indexes.push_back(data.substr(0,data.find(',')));
			}
			vector<string>cols = camma_remove(data);
			vector<string>row_datas;
			for (string col : cols) {
				row_datas.push_back(col);
			}
			datas.push_back(row_datas);
		}
		else {
			vector<string> cols = split(data, ',');
			if (hasIndexRows) {
				indexes.push_back(cols[0]);
				cols.erase(cols.begin());
			}
			vector<string>row_datas;
			for (string col : cols) {
				row_datas.push_back(col);
			}
			datas.push_back(row_datas);
		}
	}
	rows = datas.size();
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
MatrixX<T> Data<T>::to_Matrix() {
	Matrix<T, Dynamic, Dynamic>mat(rows, columns);
	int r = 0, c = 0;
	for (vector<string> row : datas) {
		for (string col : row) {
			istringstream iss(col);
			T data_t;
			iss >> data_t;
			mat(r, c) = data_t;
			c++;
		}
		r++;
		c = 0;
	}
	return mat;
}

/*template<typename T>
MatrixX<T> Data<T>::to_Matrix() {
	vector<T> dm;
	for (string data : datas) {
		istringstream iss(data);
		T data_t;
		iss >> data_t;
		dm.push_back(data_t);
	}
	Map<Matrix<T, Dynamic, Dynamic, RowMajor>>mat(dm.data(), rows, columns);
	return mat;
}*/

template<typename T>
vector<string> Data<T>::getFeatures() const {
	return features;
}

template<typename T>
vector<string> Data<T>::getIndexs() const {
	return indexes;
}

template<typename T>
vector<vector<string>> Data<T>::getData() const {
	return datas;
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