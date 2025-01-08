#include"utils.cuh"
#include<stdio.h>
#include<stdlib.h>

template void matmul<MatrixXd>(MatrixXd& a, MatrixXd& b, MatrixXd& c);
template void matmul<VectorXd>(MatrixXd& a, VectorXd& b, VectorXd& c);
template void matmul_shared<MatrixXd>(MatrixXd& a, MatrixXd& b, MatrixXd& c);
template void matmul_shared<VectorXd>(MatrixXd& a, VectorXd& b, VectorXd& c);

#define CHECK_CUDA_ERROR(val) Check_cuda_Error((val),__FILE__,__LINE__)
void Check_cuda_Error(cudaError_t error, const char* const file, const int line) {
	if (error != cudaSuccess) {
		printf("CUDA Error Occurs at %s, line %d : #%d(%s)\n", file, line, error, cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

__global__ void multi_matrix(double* as, double* bs, double* cs, int row, int col, int k) {
	int r = threadIdx.y + blockDim.y * blockIdx.y;
	int c = threadIdx.x + blockDim.x * blockIdx.x;
	if (r < row && c < col) {
		double val = 0.0;
		for (int i = 0; i < k; i++) {
			val += as[r + i * row] * bs[c * k + i];
		}
		cs[r + c * row] = val;
	}
}

template<typename T>
void matmul(MatrixXd& a, T& b, T& c) {
	//create pointer
	double* as;
	double* bs;
	double* cs;
	//size of matrix
	size_t a_sz = sizeof(double) * a.size();
	size_t b_sz = sizeof(double) * b.size();
	size_t c_sz = sizeof(double) * c.size();
	//allocate memory to pointer
	CHECK_CUDA_ERROR(cudaMalloc(&as, a_sz));
	CHECK_CUDA_ERROR(cudaMalloc(&bs, b_sz));
	CHECK_CUDA_ERROR(cudaMalloc(&cs, c_sz));
	//copy matrix data to pointer
	CHECK_CUDA_ERROR(cudaMemcpy(as, a.data(), a_sz, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(bs, b.data(), b_sz, cudaMemcpyHostToDevice));
	//threads and block size
	dim3 threads_per_block(32, 32, 1);
	dim3 blocks_per_grid((b.cols() + 32 - 1) / 32, (a.rows() + 32 - 1) / 32, 1);
	multi_matrix << <blocks_per_grid, threads_per_block >> > (as, bs, cs, a.rows(), b.cols(), a.cols());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	//copy pointer data to matrix
	CHECK_CUDA_ERROR(cudaMemcpy(c.data(), cs, c_sz, cudaMemcpyDeviceToHost));
	//release allocated memory
	CHECK_CUDA_ERROR(cudaFree(as));
	CHECK_CUDA_ERROR(cudaFree(bs));
	CHECK_CUDA_ERROR(cudaFree(cs));
	return;
}

__global__ void multi_matrix_shared(double* a,double* b, double* c, int row, int col, int k) {
	const int BM = 128;
	const int BN = 128;
	const int BK = 8;
	const int TM = 8;
	const int TN = 8;

	__shared__ double a_shared[BM][BK];//a_shared is a 128*8 matrix
	__shared__ double b_shared[BK][BN];//b_shared is a 8*128 matrix
	double sub_c[TM][TN] = { 0.0 };//sub_c is a 8*8 matrix that save part of matrix multiply by a_shared and b_shared

	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int ty = threadIdx.y;
	const int tx = threadIdx.x;
	const int tid = ty * blockDim.x + tx;//thread index
	//one thread construct part of a_shared that size is 1*4
	int a_shared_row = tid >> 1;
	int a_shared_col = (tid % 2) << 2;
	//one thread construct part of b_shared that size is 1*4
	int b_shared_row = tid >> 5;
	int b_shared_col = (tid % 32) << 2;

	int a_global_row = by * BM + a_shared_row;//the row that corresponding to pointer a's row
	int b_global_col = bx * BN + b_shared_col;//the column that corresponding to pointer b's column

	for (int bk = 0; bk < (k + BK - 1) / BK; bk++) {
		int a_global_col = bk * BK + a_shared_col;//the column that corresponding to pointer a's column
		int a_global_site = a_global_row + a_global_col * row;//the site that corresponding to pointer a's site

		int b_global_row = bk * BK + b_shared_row;//the row that corresponding to pointer b's row
		int b_global_site = b_global_row + b_global_col * k;//the site that corresponding to pointer b's site
		for (int i = 0; i < 4; i++) {//construct a_shared and b_shared matrix
			a_shared[a_shared_row][a_shared_col + i] = (a_global_row < row && a_global_col < k) ? a[a_global_site + i * row] : 0;
			b_shared[b_shared_row][b_shared_col + i] = (b_global_row < k && b_global_col < col) ? b[b_global_site + i * k] : 0;
		}
		__syncthreads();//every threads synchronous construct these two shared matrics

		#pragma unroll
		for (int j = 0; j < BK; j++) {
			#pragma unroll
			for (int m = 0; m < TM; m++) {
				int a_shared_inc_row = ty * TM + m;//the row in a_shared
				#pragma unroll
				for (int n = 0; n < TN; n++) {//calculate sub_c matrix
					int b_shared_inc_col = tx * TN + n;//the column in b_shared
					sub_c[m][n] += a_shared[a_shared_inc_row][j] * b_shared[j][b_shared_inc_col];//multiply two matrics
				}
			}
		}
		__syncthreads();//every threads synchronous multiply part of these two shared matrics to get 8*8 sub_c matrix
	}
	#pragma unroll
	for (int m = 0; m < TM; m++) {
		int c_global_row = by * BM + ty * TM + m;//the row that corresponding to pointer c's row
		if (c_global_row >= row) { break; }
		#pragma unroll
		for (int n = 0; n < TN; n += 4) {
			int c_global_col = bx * BN + tx * TN + n;//the column that corresponding to pointer c's column
			if (c_global_col >= col) { break; }
			int c_global_site = c_global_row + c_global_col * row; //the site that corresponding to pointer c's site
			for (int i = 0; i < 4; i++) {
				if (c_global_col + i >= col) { break; }
				c[c_global_site + i * row] = sub_c[m][n + i];//save the result of sub_c to pointer c
			}
		}
	}
}

template<typename T>
void matmul_shared(MatrixXd& a, T& b, T& c) {
	//create pointer
	double* aptr;
	double* bptr;
	double* cptr;
	//size of matrix
	size_t a_size = sizeof(double) * a.size();
	size_t b_size = sizeof(double) * b.size();
	size_t c_size = sizeof(double) * c.size();
	//allocate memory to pointer
	CHECK_CUDA_ERROR(cudaMalloc(&aptr, a_size));
	CHECK_CUDA_ERROR(cudaMalloc(&bptr, b_size));
	CHECK_CUDA_ERROR(cudaMalloc(&cptr, c_size));
	//copy matrix data to pointer
	CHECK_CUDA_ERROR(cudaMemcpy(aptr, a.data(), a_size, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(bptr, b.data(), b_size, cudaMemcpyHostToDevice));
	//thread and block size
	dim3 blocksdim(16, 16, 1);
	dim3 gridsdim((b.cols() - 1) / 128 + 1, (a.rows() - 1) / 128 + 1, 1);
	multi_matrix_shared << <gridsdim, blocksdim >> > (aptr, bptr, cptr, a.rows(), b.cols(), a.cols());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	//copy pointer data to matrix
	CHECK_CUDA_ERROR(cudaMemcpy(c.data(), cptr, c_size, cudaMemcpyDeviceToHost));
	//release allocate memory
	CHECK_CUDA_ERROR(cudaFree(aptr));
	CHECK_CUDA_ERROR(cudaFree(bptr));
	CHECK_CUDA_ERROR(cudaFree(cptr));
	return;
}


__global__ void multi_matvec_shared(double* a, double* b, double* c, int row, int k) {

	const int BK = 512;
	const int TM = 128;

	__shared__ double b_shared[BK];
	double sub_c[TM] = { 0.0 };

	const int bx = blockIdx.x;
	const int by = blockIdx.y;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	int a_global_row = 8 * by * blockDim.y + 8 * ty;

	for (int bk = 0; bk < (k - 1) / BK + 1; bk++) {
		for (int i = 0; i < 8; i++) {
			b_shared[8 * tx + i] = (bk * BK + 8 * tx + i < k) ? b[bk * BK + 8 * tx + i] : 0;
		}
		__syncthreads();
		#pragma unroll
		for (int j = 0; j < BK; j++) {
			if (j + bk * BK >= k) { break; }
			#pragma unroll
			for (int i = 0; i < 8; i++) {
				if (a_global_row + i >= row) { break; }
				sub_c[8 * ty + i] += a[a_global_row + i + row * (j + bk * BK)] * b_shared[j];
			}
		}
		__syncthreads();
	}
	int c_global_row = 8 * by * blockDim.y + 8 * ty;
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		if (c_global_row + i >= row) { break; }
		c[c_global_row + i] = sub_c[8 * ty + i];
	}
}

void matvecmul_shared(MatrixXd& a, VectorXd& b, VectorXd& c) {
	double* aptr;
	double* bptr;
	double* cptr;
	size_t a_size = sizeof(double) * a.size();
	size_t b_size = sizeof(double) * b.size();
	size_t c_size = sizeof(double) * c.size();
	CHECK_CUDA_ERROR(cudaMalloc(&aptr, a_size));
	CHECK_CUDA_ERROR(cudaMalloc(&bptr, b_size));
	CHECK_CUDA_ERROR(cudaMalloc(&cptr, c_size));
	CHECK_CUDA_ERROR(cudaMemcpy(aptr, a.data(), a_size, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(bptr, b.data(), b_size, cudaMemcpyHostToDevice));
	dim3 blocksdim(64, 16, 1);
	dim3 gridsdim((a.cols() - 1) / 512 + 1, (a.rows() - 1) / 128 + 1, 1);
	multi_matvec_shared << <gridsdim, blocksdim >> > (aptr, bptr, cptr, a.rows(), a.cols());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	CHECK_CUDA_ERROR(cudaMemcpy(c.data(), cptr, c_size, cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaFree(aptr));
	CHECK_CUDA_ERROR(cudaFree(bptr));
	CHECK_CUDA_ERROR(cudaFree(cptr));
	return;
}

__global__ void add_matrix(double* a, double* b, double* c, int row, int col) {
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < col && y < row) {
		c[y + x * row] = a[y + x * row] + b[y + x * row];
	}
}

void matadd(MatrixXd& a, MatrixXd& b, MatrixXd& c) {
	double* aptr;
	double* bptr;
	double* cptr;
	size_t a_size = sizeof(double) * a.size();
	CHECK_CUDA_ERROR(cudaMalloc(&aptr, a_size));
	CHECK_CUDA_ERROR(cudaMalloc(&bptr, a_size));
	CHECK_CUDA_ERROR(cudaMalloc(&cptr, a_size));
	CHECK_CUDA_ERROR(cudaMemcpy(aptr, a.data(), a_size, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(bptr, b.data(), a_size, cudaMemcpyHostToDevice));
	dim3 blockdim(32, 32, 1);
	dim3 griddim((a.cols() - 1) / 32 + 1, (a.rows() - 1) / 32 + 1, 1);
	add_matrix << <griddim, blockdim >> > (aptr, bptr, cptr, a.rows(), a.cols());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	CHECK_CUDA_ERROR(cudaMemcpy(c.data(), cptr, a_size, cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaFree(aptr));
	CHECK_CUDA_ERROR(cudaFree(bptr));
	CHECK_CUDA_ERROR(cudaFree(cptr));
	return;
}