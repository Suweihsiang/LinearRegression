#include"data.h"

template<typename T>
Data<T>::Data(unordered_map<string, vector<string>>m, bool IndexFirst) :rows((m.begin()->second).size()), columns(m.size()), hasIndexRows(IndexFirst) {
	auto it = m.begin();
	if (hasIndexRows) {
		index_name = it->first;
		indexes = it->second;
		mat.conservativeResize(rows, columns-1);
		for (int c = 1; c < columns; c++) {
			it++;
			features.push_back(it->first);
			for (int r = 0; r < rows; r++) {
				string s_val = it->second[r];
				istringstream iss(s_val);
				T T_val;
				iss >> T_val;
				mat(r, c-1) = T_val;
			}
		}
	}
	else {
		mat.conservativeResize(rows, columns);
		for (int c = 0; c < columns; c++) {
			features.push_back(it->first);
			for (int r = 0; r < rows; r++) {
				string s_val = it->second[r];
				istringstream iss(s_val);
				T T_val;
				iss >> T_val;
				mat(r, c) = T_val;
			}
			it++;
		}
	}
	if (IndexFirst && *indexes.begin() > *(indexes.end() - 1)) {
		reverse(indexes.begin(), indexes.end());
		mat = mat.colwise().reverse().eval();
	}
}

template<typename T>
Data<T>::Data(string path,bool IndexFirst,bool isThousand) {
	istringstream namess(path);
	getline(namess, data_name, '.');
	data_name = data_name.substr(data_name.rfind('/')+1, data_name.size());
	ifstream ifs(path);
	string data;
	hasIndexRows = IndexFirst;
	getline(ifs, data);
	features = split(data, ',');
	index_name = features[0];
	features.erase(features.begin());
	columns = features.size();
	mat.conservativeResize(0, columns);
	int r = 0, c = 0;
	while (getline(ifs, data)) {
		mat.conservativeResize(mat.rows() + 1, columns);
		if (isThousand) {
			if (hasIndexRows) {
				indexes.push_back(data.substr(0,data.find(',')));
				vector<string>cols = camma_remove(data);
				for (string col : cols) {
					istringstream iss(col);
					T T_val;
					iss >> T_val;
					mat(r, c) = T_val;
					c++;
				}
			}
			else {
				istringstream is(data.substr(0, data.find(',')));
				T T_date;
				is >> T_date;
				mat(r, c) = T_date;
				c++;
				vector<string>cols = camma_remove(data);
				for (string col : cols) {
					istringstream iss(col);
					T T_val;
					iss >> T_val;
					mat(r, c) = T_val;
					c++;
				}
			}
		}
		else {
			vector<string> cols = split(data, ',');
			if (hasIndexRows) {
				indexes.push_back(cols[0]);
				cols.erase(cols.begin());
				for (string col : cols) {
					istringstream iss(col);
					T T_val;
					iss >> T_val;
					mat(r, c) = T_val;
					c++;
				}
			}
			else {
				for (string col : cols) {
					istringstream iss(col);
					T T_val;
					iss >> T_val;
					mat(r, c) = T_val;
					c++;
				}
			}
		}
		r++;
		c = 0;
	}
	if (IndexFirst && *indexes.begin() > *(indexes.end() - 1)) {
		reverse(indexes.begin(), indexes.end());
		mat = mat.colwise().reverse().eval();
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

template<typename T>
void Data<T>::removeRow(int RowToRemove) {
	mat.block(RowToRemove, 0, mat.rows() - RowToRemove-1, mat.cols()) = mat.block(RowToRemove + 1, 0, mat.rows() - RowToRemove-1, mat.cols());
	mat.conservativeResize(mat.rows() - 1, mat.cols());
}

template<typename T>
void Data<T>::removeRows(vector<string>idxs) {
	for (string idx : idxs) {
		auto it = find(indexes.begin(), indexes.end(), idx);
		if (it != indexes.end()) {
			auto next = indexes.erase(it);
			int dist = distance(indexes.begin(), next);
			mat.block(dist, 0, mat.rows() - dist - 1, mat.cols()) = mat.block(dist + 1, 0, mat.rows() - dist - 1, mat.cols());
			mat.conservativeResize(mat.rows() - 1, mat.cols());
			rows--;
		}
	}
}

template<typename T>
void Data<T>::merge(Data df2) {
	vector<string> features2 = df2.getFeatures();
	vector<string> indexes2 = df2.getIndexs();
	for (string feature : features2) {
		if (find(features.begin(),features.end(),feature) == features.end()) {
			features.push_back(feature);
		}
		else {
			features.push_back(feature + "_" + df2.data_name);
		}
	}
	int remove_count = 0;
	for (int i = 0; i < df2.getRows(); i++) {
		if (find(indexes.begin(),indexes.end(),indexes2[i]) == indexes.end()) {
			df2.removeRow(i-remove_count);
			remove_count++;
		}
	}
	remove_count = 0;
	vector<int>rm_idx;
	for (int i = 0; i < getRows(); i++) {
		if (find(indexes2.begin(), indexes2.end(),indexes[i]) == indexes2.end()) {
			removeRow(i-remove_count);
			rm_idx.push_back(i-remove_count);
			remove_count++;
		}
	}
	for (int idx : rm_idx) {
		auto it = indexes.erase(indexes.begin() + idx);
	}
	rows = indexes.size();
	columns = features.size();
	Matrix<T, Dynamic, Dynamic> mat2 = df2.getMatrix();
	mat.conservativeResize(mat.rows(), mat.cols() + mat2.cols());
	mat.block(0, mat.cols() - mat2.cols(), mat2.rows(), mat2.cols()) = mat2;
}

template<typename T>
void Data<T>::addRows(vector<vector<string>>addrows) {
	size_t r = addrows.size();
	size_t c = addrows[0].size();
	mat.conservativeResize(mat.rows() + r, mat.cols());
	if (hasIndexRows) {
		for (int i = 0; i < r; i++) {
			indexes.push_back(addrows[i][0]);
			for (int j = 1; j < c; j++) {
				istringstream iss(addrows[i][j]);
				T T_val;
				iss >> T_val;
				mat(mat.rows() - r + i, j - 1) = T_val;
			}
		}
	}
	else {
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				istringstream iss(addrows[i][j]);
				T T_val;
				iss >> T_val;
				mat(mat.rows() - r + i, j) = T_val;
			}
		}
	}
	rows += r;
}

template<typename T>
void Data<T>::removeColumns(vector<string>fts) {
	for (string ft : fts) {
		auto it = find(features.begin(), features.end(), ft);
		if (it != features.end()) {
			auto next = features.erase(it);
			int dist = distance(features.begin(), next);
			mat.block(0, dist, rows, columns - dist - 1) = mat.block(0, dist, rows, columns - dist - 1);
			mat.conservativeResize(mat.rows(), mat.cols() - 1);
			columns--;
		}
	}
}

template<typename T>
void Data<T>::print() {
	cout << index_name << "\t"<<" ";
	for (string feature : features) {
		cout << feature << "\t"<<" ";
	}
	cout << endl;
	for (int i = 0; i < rows; i++) {
		cout << indexes[i] << " ";
		cout << mat.row(i) << endl;
	}
}