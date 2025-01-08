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
#include<set>

using std::unordered_map;
using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;
using std::ios;
using std::istringstream;
using std::set;
using std::pair;
using std::cout;
using std::endl;
using std::to_string;


using Eigen::Matrix;
using Eigen::MatrixX;
using Eigen::Dynamic;
using Eigen::RowVectorX;
using Eigen::VectorX;

template<typename T>
class Data {
public:
	Data();														  //constructor
	Data(unordered_map<string, vector<string>>m, bool IndexFirst);//constructed by unordered_map
	Data(string path,bool IndexFirst,bool isThousand);			  //constructed by file path
	~Data();													  //destructor
	Data<T> operator[](vector<string> fts);						  //select features
	vector<string> split(string s, char dec);					  //split string s by dec
	void setDataname(string _data_name);						  //set data name
	string getDataname() const;									  //get data name
	int getRows() const;										  //get number of rows
	int getColumns() const;										  //get number of columns
	Matrix<T, Dynamic, Dynamic> getMatrix() const;				  //get matrix
	vector<string> getFeatures() const;							  //get data's features
	vector<string>getIndexs() const;							  //get datas indexes
	vector<string> camma_remove(string data_string);			  //remove data's camma
	void setIndex(string index);								  //set feature as data index
	void removeRow(int RowToRemove);							  //remove row by number
	void removeRow(string idx);									  //remove row by index
	void removeRows(vector<string>idxs);						  //remove some indexes
	void merge(Data df2);										  //merge another data to this data
	void addRows(vector<vector<string>>rows);				      //add some row datas
	void addColumns(unordered_map<string, vector<string>>m);	  //add features
	void removeColumn(int ColToRemove);							  //remove column by number of column
	void removeColumns(vector<string>fts);						  //remove some columns by feature's name
	void renameColumns(unordered_map<string, string>names);		  //rename some features' name
	void sortbyIndex(bool ascending);							  //sort data by index
	void sortby(string feature,bool ascedning);					  //sort data by feature
	Data<T> groupby(string feature, string operate);			  //group data by some feature
	void dropna(int axis);										  //drop null axis
	void print();												  //print data
	void to_csv(string path);									  //save data to csv file
private:
	string data_name;
	int rows = 0;
	int columns = 0;
	bool hasIndexRows;
	string index_name;
	vector<string>features;
	vector<string>indexes;
	vector<vector<int>>nullpos;
	Matrix<T,Dynamic,Dynamic>mat;
	void setMatrix(unordered_map<string, vector<string>>::iterator it);
	void setMatrix(int r, int c, vector<string> cols);
};

#endif