#include"data.h"
//declare the template variable type
template Data<double>;
template Data<float>;
template Data<int>;

//implement
template<typename T>
Data<T>::Data() {}

template<typename T>
Data<T>::Data(unordered_map<string, vector<string>>m, bool IndexFirst) :rows((m.begin()->second).size()), hasIndexRows(IndexFirst) {
	auto it = m.begin();
	columns = hasIndexRows ? m.size() - 1 : m.size();//how many features?
	mat.conservativeResize(rows, columns);//reserve data matrix size
	if (hasIndexRows) {//set index
		index_name = it->first;
		indexes = it->second;
		it++;
	}
	setMatrix(it);//set matrix
	sortbyIndex(true);//sort
}

template<typename T>
Data<T>::Data(string path,bool IndexFirst,bool isThousand) {
	data_name = path.substr(path.rfind('/') + 1, path.rfind('.') - path.rfind('/') - 1);//set data name
	ifstream ifs(path);
	string data;
	hasIndexRows = IndexFirst;
	getline(ifs, data);
	if (hasIndexRows) {
		features = split(data, ',');//get features vector
		index_name = features[0];//first feature is index
		features.erase(features.begin());//first feature is index, so erase it from features vector
	}
	else {
		features = split(data, ',');//get features vector
	}
	columns = features.size();//how many features
	mat.conservativeResize(0, columns);//reserve matrix size
	int r = 0, c = 0;
	while (getline(ifs, data)) {//start at second row
		vector<string>cols;
		mat.conservativeResize(mat.rows() + 1, columns);//resize matrix row
		if (isThousand) {//data has camma
			if (hasIndexRows) {
				indexes.push_back(data.substr(0,data.find(',')));//set data index
			}
			else {
				istringstream is(data.substr(0, data.find(',')));
				T T_date;//generic data
				is >> T_date;//string to generic type
				mat(r, c) = T_date;//set mat(r,c) data
				c++;//next column
			}
			cols = camma_remove(data);//remove camma
		}
		else {
			cols = split(data, ',');//get datas vector
			if (hasIndexRows) {
				indexes.push_back(cols[0]);
				cols.erase(cols.begin());
			}
		}
		setMatrix(r, c, cols);//set data matrix
		r++;//next row
		c = 0;//back to the first column
	}
	rows = mat.rows();//set total number of data
	sortbyIndex(true);//sort data
}

template<typename T>
Data<T>::~Data() {}

template<typename T>
Data<T> Data<T>::operator[](vector<string>fts) {
	Data<T> d_fts;
	//set selected data basic property
	d_fts.data_name = data_name;
	d_fts.hasIndexRows = hasIndexRows;
	if (hasIndexRows) {
		d_fts.index_name = index_name;
		d_fts.indexes = indexes;
	}
	//
	int c = 0;
	for (const auto &ft : fts) {
		auto it = find(features.begin(), features.end(), ft);//iterator of selected features
		if (it != features.end()) {
			d_fts.mat.conservativeResize(rows, c + 1);//resize matrix column
			d_fts.features.push_back(ft);//add feature
			size_t dist = distance(features.begin(), it);//number of column that feature in
			d_fts.mat.col(c) = mat.col(dist);//set data matrix column
			c++;//next feature
		}
	}
	d_fts.rows = rows;//number of rows of selected data
	d_fts, columns = d_fts.mat.cols();//number of columns of selected data
	return d_fts;
}

template<typename T>
void Data<T>::setMatrix(unordered_map<string, vector<string>>::iterator it) {//set matrix by unordered_map
	for (int c = 0; c < columns; c++) {//set matrix feature by feature
		features.push_back(it->first);//set feature
		for (int r = 0; r < rows; r++) {
			string s_val = it->second[r];
			if (s_val == "") {//if data is null
				nullpos.push_back({ r,c });//record this place
				c++;//next column
				continue;
			}
			istringstream iss(s_val);
			T T_val;//generic type
			iss >> T_val;//string to generic type
			mat(r, c) = T_val;//set this index
		}
		it++;//next feature
	}
}

template<typename T>
void Data<T>::setMatrix(int r, int c, vector<string>cols) {
	for (const auto &col : cols) {
		if (col == "") {
			nullpos.push_back({ r,c });
			c++;
			continue;
		}
		istringstream iss(col);
		T T_val;
		iss >> T_val;
		mat(r, c) = T_val;
		c++;
	}
	while (c < columns) {
		nullpos.push_back({ r,c });
		c++;
	}
}

template<typename T>
vector<string> Data<T>::split(string s, char dec) {//split string s by dec
	istringstream iss(s);
	string subs;
	vector<string>sv;
	while (getline(iss, subs, dec)) {
		sv.push_back(subs);
	}
	return sv;
}

template<typename T>
void Data<T>::setDataname(string _data_name) {
	data_name = _data_name;
}

template<typename T>
string Data<T>::getDataname() const {
	return data_name;
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
		begin = data.find('"');//find first " of this data
		end = data.substr(begin + 1).find('"');//find last " of this data
		string nd = data.substr(begin+1, end);//remove first and last "
		do {//processed this data
			nd.erase(nd.find(','),1);//remove camma
		} while (nd.find(',') != nd.npos);
		numeric_data.push_back(nd);//add data that has been processed
		data = data.substr(begin + end + 2);//next data
	} while (data.find('"') != data.npos);
	return numeric_data;
}

template<typename T>
void Data<T>::setIndex(string index) {
	auto it = find(features.begin(), features.end(), index);//this feature's iterator
	if (it != features.end()) {
		size_t dist = distance(features.begin(), it);//this feature's position
		if (hasIndexRows) {//has set first feature as index when constructed
			mat.conservativeResize(rows, columns + 1);//add column for the origin index
			features.push_back(index_name);//put the origin index to the feature vector
			for (int i = 0; i < rows; i++) {//put the origin index's data to data matrix
				istringstream iss(indexes[i]);
				T T_val;
				iss >> T_val;
				mat(i, columns) = T_val;
			}
			columns++;
			vector<T> new_indexes(mat.col(dist).data(),mat.col(dist).data() + mat.rows());//new index vector
			removeColumns({ index });//remove new index from feature vector
			index_name = index;
			indexes.clear();
			for (const auto &idx : new_indexes) {
				indexes.push_back(to_string((int)idx));
			}
		}
		else {
			hasIndexRows = true;
			vector<T> new_indexes(mat.col(dist).data(),mat.col(dist).data() + mat.rows());
			removeColumns({ index });
			index_name = index;
			for (const auto &idx : new_indexes) {
				indexes.push_back(to_string((int)idx));
			}
		}
	}
}

template<typename T>
void Data<T>::removeRow(int RowToRemove) {
	indexes.erase(indexes.begin() + RowToRemove);//remove the row index
	mat.block(RowToRemove, 0, rows - RowToRemove - 1, columns) = mat.block(RowToRemove + 1, 0, rows - RowToRemove - 1, columns);//remove the row from matrix
	mat.conservativeResize(rows - 1, columns);//resize matrix
	rows--;
}

template<typename T>
void Data<T>::removeRow(string idx) {
	auto it = find(indexes.begin(), indexes.end(), idx);//the index's iterator
	if (it != indexes.end()) {
		auto next = indexes.erase(it);//erase the index
		int dist = distance(indexes.begin(), next);//the index's row
		mat.block(dist, 0, mat.rows() - dist - 1, mat.cols()) = mat.block(dist + 1, 0, mat.rows() - dist - 1, mat.cols());//remove this row from the matrix
		mat.conservativeResize(mat.rows() - 1, mat.cols());//resize matrix
		rows--;
	}
}

template<typename T>
void Data<T>::removeRows(vector<string>idxs) {
	for (const auto &idx : idxs) {//remove indexes one by one
		removeRow(idx);
	}
}

template<typename T>
void Data<T>::merge(Data df2) {
	vector<string> features2 = df2.getFeatures();//second data's features
	vector<string> indexes2 = df2.getIndexs();//second data's indexes
	for (const auto &feature : features2) {
		if (find(features.begin(),features.end(),feature) == features.end()) {//there is a feature in second data but not in origin data
			features.push_back(feature);
		}
		else {
			features.push_back(feature + "_" + df2.data_name);//distinguish the same feature name from two datas
		}
	}
	for (const auto &idx : df2.getIndexs()) {
		if (find(indexes.begin(),indexes.end(),idx) == indexes.end()) {//there is a index in second data but not in origin data
			df2.removeRow(idx);//remove the index from second data
		}
	}
	for (const auto &idx : getIndexs()) {
		if (find(indexes2.begin(), indexes2.end(),idx) == indexes2.end()) {//there is a index in origin data but not in second data
			removeRow(idx);//remove the index from origin data
		}
	}
	columns = features.size();//new feature size
	Matrix<T, Dynamic, Dynamic> mat2 = df2.getMatrix();
	mat.conservativeResize(mat.rows(), mat.cols() + mat2.cols());//resize matrix size
	mat.block(0, mat.cols() - mat2.cols(), mat2.rows(), mat2.cols()) = mat2;//add second data's matrix to mat
}

template<typename T>
void Data<T>::addRows(vector<vector<string>>addrows) {
	size_t r = addrows.size();
	size_t c = addrows[0].size();
	mat.conservativeResize(mat.rows() + r, mat.cols());//resize matrix
	//add index(if necessary) and row data
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
	//
	rows += r;//the number of new rows
}

template<typename T>
void Data<T>::addColumns(unordered_map<string, vector<string>>ms) {
	auto it = ms.begin();
	for (int c = 0; c < ms.size(); c++) {
		mat.conservativeResize(rows, columns + 1);//resize matrix
		features.push_back(it->first);//add new feature
		for (int r = 0; r < rows; r++) {//add new feature datas
			string s_val = it->second[r];
			istringstream iss(s_val);
			T T_val;
			iss >> T_val;
			mat(r, columns) = T_val;
		}
		it++;//next new feature
		columns++;//new number of columns
	}
}

template<typename T>
void Data<T>::removeColumn(int ColToRemove) {
	features.erase(features.begin() + ColToRemove);//erase from feature vector
	mat.block(0, ColToRemove, rows, columns - ColToRemove - 1) = mat.block(0, ColToRemove + 1, rows, columns - ColToRemove - 1);//erase from matrix
	mat.conservativeResize(rows, columns - 1);//resize matrix
	columns--;
}

template<typename T>
void Data<T>::removeColumns(vector<string>fts) {
	for (const auto &ft : fts) {
		auto it = find(features.begin(), features.end(), ft);//removed feature's iterator
		if (it != features.end()) {
			auto next = features.erase(it);//erase this feature from vector
			int dist = distance(features.begin(), next);//this feature's column
			mat.block(0, dist, rows, columns - dist - 1) = mat.block(0, dist+1, rows, columns - dist - 1);//remove the column from matrix
			mat.conservativeResize(mat.rows(), mat.cols() - 1);//resize matrix
			columns--;
		}
	}
}

template<typename T>
void Data<T>::renameColumns(unordered_map<string, string>names) {//map{old_name,new_name}
	for (const auto& name : names) {
		vector<string>::iterator it = find(features.begin(), features.end(), name.first);//find old feature name
		if (it != features.end()) {
			int idx = distance(features.begin(), it);//old feature name's position
			features[idx] = name.second;//rename it
		}
	}
}

template<typename T>
void Data<T>::sortbyIndex(bool ascending) {
	if (!hasIndexRows) {
		return;
	}
	vector<pair<string, VectorX<T>>>vec;
	for (int i = 0; i < rows; i++) {
		vec.push_back({ indexes[i],mat.row(i) });//zip the index and data
	}
	if (ascending) {
		sort(vec.begin(), vec.end(), [](pair<string, VectorX<T>>& v1, pair<string, VectorX<T>>& v2) {return v1.first < v2.first; });
	}
	else {
		sort(vec.begin(), vec.end(), [](pair<string, VectorX<T>>& v1, pair<string, VectorX<T>>& v2) {return v1.first > v2.first; });
	}
	for (int i = 0; i < rows; i++) {//unzip the index and data
		indexes[i] = vec[i].first;
		mat.row(i) = vec[i].second;
	}
}

template<typename T>
void Data<T>::sortby(string feature,bool ascending) {
	if (feature == index_name) {
		sortbyIndex(ascending);
	}
	else {
		auto it = find(features.begin(), features.end(), feature);
		int dist = distance(features.begin(), it);//the feature's position
		vector<pair<string, VectorX<T>>>vec;
		for (int i = 0; i < rows; i++) {
			hasIndexRows?vec.push_back({ indexes[i],mat.row(i) }): vec.push_back({ to_string(i+1),mat.row(i)});
		}
		if (ascending) {
			sort(vec.begin(), vec.end(), [&dist](pair<string, VectorX<T>>& v1, pair<string, VectorX<T>>& v2) {return v1.second[dist] < v2.second[dist]; });
		}
		else {
			sort(vec.begin(), vec.end(), [&dist](pair<string, VectorX<T>>& v1, pair<string, VectorX<T>>& v2) {return v1.second[dist] > v2.second[dist]; });
		}
		for (int i = 0; i < rows; i++) {
			if (hasIndexRows) { indexes[i] = vec[i].first; }
			mat.row(i) = vec[i].second;
		}
	}
}

template<typename T>
Data<T> Data<T>::groupby(string feature, string operate) {
	setIndex(feature);//set the feature as index
	sortbyIndex(true);
	set<string> idx_set(indexes.begin(), indexes.end());//set index uniquely
	MatrixX<T> mat_res(0, columns);
	int pos = 0;//current row
	for (const auto &idx : idx_set) {
		RowVectorX<T> sum = RowVectorX<T>::Zero(columns);//set this group's row data all zeros
		int count = 0;//reset this group's count
		for (int i = pos; i < rows; i++) {
			if (indexes[i] == idx) {//this index is belong to this group
				sum += mat.row(i);
				count++;
				pos++;
			}
			else {
				break;
			}
		}
		if (operate == "sum") { ; }
		else if (operate == "mean") { sum /= count; }
		mat_res.conservativeResize(mat_res.rows() + 1, columns);
		mat_res.row(mat_res.rows()-1) = sum;//this group's row data
	}
	//prepare for the constructor of group data
	vector<string>idx_group(idx_set.begin(), idx_set.end());//index of group data
	vector<vector<string>>res_group;
	for (int c = 0; c < columns; c++) {
		vector<string>mat_to_str;
		for (int r = 0;r < mat_res.rows(); r++) {
			mat_to_str.push_back(to_string(mat_res(r, c)));
		}
		res_group.push_back(mat_to_str);
	}
	unordered_map<string, vector<string>>mav;//construct using unordered_map 
	mav[index_name] = idx_group;
	for (int i = 0; i < columns; i++) {
		mav[features[i]] = res_group[i];
	}
	Data<T> df_group(mav, true);//constructor of group data
	return df_group;
}

template<typename T>
void Data<T>::dropna(int axis) {//axis = 0 is row, 1 is column
	sort(nullpos.begin(), nullpos.end(), [&axis](vector<int>& v1, vector<int>& v2) {return v1[axis] < v2[axis]; });
	int removeCount = 0;
	int pre_remove = -1;
	for (const auto &nan : nullpos) {
		if (nan[axis] != pre_remove) {//the axis has not been removed
			(axis == 0) ? removeRow(nan[axis] - removeCount) : removeColumn(nan[axis] - removeCount);
			removeCount++;//how many axis has been removed
			pre_remove = nan[axis];
		}
	}

}

template<typename T>
void Data<T>::print() {
	if (hasIndexRows) {
		cout << index_name << "\t" << " ";
	}
	for (const auto &feature : features) {
		cout << feature << "\t"<<" ";
	}
	cout << endl;
	for (int i = 0; i < rows; i++) {
		if (hasIndexRows) {
			cout << indexes[i] << " ";
		}
		cout << mat.row(i) << endl;
	}
}

template<typename T>
void Data<T>::to_csv(string path) {
	ofstream dataFile;
	dataFile.open(path, ios::out | ios::trunc);
	if (hasIndexRows) {
		dataFile << index_name << " ";
	}
	for (const auto &feature : features) {
		dataFile << feature << " ";
	}
	dataFile << endl;
	for (int r = 0; r < rows; r++) {
		if (hasIndexRows) {
			dataFile << indexes[r] << " ";
		}
		dataFile<< mat.row(r);
		dataFile << endl;
	}
	dataFile.close();
}