#ifndef UTILS_CUH
#define UTILS_CUH

#include<eigen3/Eigen/Dense>
#include<eigen3/Eigen/Core>
#include"cuda_runtime.h"
#include"device_launch_parameters.h"

using Eigen::MatrixXd;

void matmul(MatrixXd& a, MatrixXd& b, MatrixXd& c);
void matmul_shared(MatrixXd &a, MatrixXd &b,MatrixXd &c);
void matadd(MatrixXd &a, MatrixXd &b,MatrixXd &c);

#endif