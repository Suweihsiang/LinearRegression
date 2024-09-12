#include"data.h"

Data::Data(string path,bool FeatureFirst,bool IndexFirst,bool isThousand) {
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
			for (string col : cols) {
				datas.push_back(col);
			}
		}
		else {
			vector<string> cols = split(data, ',');
			if (hasIndexRows) {
				indexes.push_back(cols[0]);
				cols.erase(cols.begin());
			}
			for (string col : cols) {
				datas.push_back(col);
			}
		}
	}
	rows = datas.size() / columns;
}

vector<string> Data::split(string s, char dec) {
	istringstream iss(s);
	string subs;
	vector<string>sv;
	while (getline(iss, subs, dec)) {
		sv.push_back(subs);
	}
	return sv;
}

int Data::getRows() const {
	return rows;
}

int Data::getColumns() const {
	return columns;
}

template<typename T>
MatrixX<T> Data::to_Matrix() {
	vector<T> dm;
	for (string data : datas) {
		istringstream iss(data);
		T data_t;
		iss >> data_t;
		dm.push_back(data_t);
	}
	Map<Matrix<T, Dynamic, Dynamic, RowMajor>>mat(dm.data(), rows, columns);
	return mat;
}

vector<string> Data::getFeatures() const {
	return features;
}

vector<string> Data::getIndexs() const {
	return indexes;
}

vector<string> Data::getData() const {
	return datas;
}

vector<string> Data::camma_remove(string data) {
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