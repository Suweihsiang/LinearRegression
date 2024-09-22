#ifndef DATA_H
#define DATA_H
#include<iostream>
#include<eigen3/Eigen/Dense>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>
#include<algorithm>
#include<unordered_map>

using namespace::std;
using namespace Eigen;

template<typename T>
class Data {
public:
	Data(unordered_map<string, vector<string>>m, bool IndexFirst);
	Data(string path,bool IndexFirst,bool isThousand);
	vector<string> split(string s, char dec);
	int getRows() const;
	int getColumns() const;
	Matrix<T, Dynamic, Dynamic> getMatrix() const;
	vector<string> getFeatures() const;
	vector<string>getIndexs() const;
	vector<string> camma_remove(string data_string);
	void removeRow(int RowToRemove);
	void removeRows(vector<string>idxs);
	void merge(Data df2);
	void addRows(vector<vector<string>>rows);
	void removeColumns(vector<string>fts);
	void print();
	void to_csv(string path);
private:
	string data_name;
	int rows = 0;
	int columns = 0;
	bool hasIndexRows;
	string index_name;
	vector<string>features;
	vector<string>indexes;
	Matrix<T,Dynamic,Dynamic>mat;
};

#endif