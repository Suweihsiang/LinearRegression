#ifndef UTILS_CUH
#define UTILS_CUH

#include<eigen3/Eigen/Dense>
#include<eigen3/Eigen/Core>
#include"cuda_runtime.h"
#include"device_launch_parameters.h"

using Eigen::MatrixXd;

cudaError_t matmul(MatrixXd& a, MatrixXd& b, MatrixXd& c);
cudaError_t matmul_shared(MatrixXd &a, MatrixXd &b,MatrixXd &c);
cudaError_t matadd(MatrixXd &a, MatrixXd &b,MatrixXd &c);

#endif