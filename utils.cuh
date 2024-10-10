#ifndef UTILS_CUH
#define UTILS_CUH

#include<eigen3/Eigen/Dense>
#include<eigen3/Eigen/Core>
#include"cuda_runtime.h"
#include"device_launch_parameters.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

template<typename T>
void matmul(MatrixXd& a, T& b, T& c);
void matmul_shared(MatrixXd &a, MatrixXd &b,MatrixXd &c);
void matvecmul_shared(MatrixXd& a, VectorXd& b, VectorXd& c);
void matadd(MatrixXd &a, MatrixXd &b,MatrixXd &c);

#endif