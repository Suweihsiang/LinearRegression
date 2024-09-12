#ifndef DATA_H
#define DATA_H
#include<iostream>
#include<eigen3/Eigen/Dense>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>
#include<algorithm>

using namespace::std;
using namespace Eigen;

template<typename T>
class Data {
public:
	Data(string path,bool FeatureFirst,bool IndexFirst,bool isThousand);
	vector<string> split(string s, char dec);
	int getRows() const;
	int getColumns() const;
	vector<string> getFeatures() const;
	vector<string>getIndexs() const;
	vector<string>getData() const;
	MatrixX<T> to_Matrix();
	vector<string> camma_remove(string data_string);
private:
	int rows = 0;
	int columns = 0;
	bool hasFeatureColumns;
	bool hasIndexRows;
	vector<string>features;
	vector<string>indexes;
	vector<string>datas;
};

#endif