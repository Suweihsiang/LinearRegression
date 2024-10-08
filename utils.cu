#include"utils.cuh"

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

cudaError_t matmul(MatrixXd& a, MatrixXd& b, MatrixXd& c) {
	double* as;
	double* bs;
	double* cs;
	size_t a_sz = sizeof(double) * a.size();
	size_t b_sz = sizeof(double) * b.size();
	size_t c_sz = sizeof(double) * c.size();
	cudaError_t cudaError;
	cudaError = cudaMalloc(&as, a_sz);
	cudaError = cudaMalloc(&bs, b_sz);
	cudaError = cudaMalloc(&cs, c_sz);
	cudaError = cudaMemcpy(as, a.data(), a_sz, cudaMemcpyHostToDevice);
	cudaError = cudaMemcpy(bs, b.data(), b_sz, cudaMemcpyHostToDevice);
	dim3 threads_per_block(32, 32, 1);
	dim3 blocks_per_grid((b.cols() + 32 - 1) / 32, (a.rows() + 32 - 1) / 32, 1);
	multi_matrix << <blocks_per_grid, threads_per_block >> > (as, bs, cs, a.rows(), b.cols(), a.cols());
	cudaError = cudaGetLastError();
	cudaError = cudaDeviceSynchronize();
	cudaError = cudaMemcpy(c.data(), cs, c_sz, cudaMemcpyDeviceToHost);
	cudaError = cudaFree(as);
	cudaError = cudaFree(bs);
	cudaError = cudaFree(cs);
	return cudaError;
}

__global__ void multi_matrix_shared(double* a,double* b, double* c, int row, int col, int k) {
	const int BM = 128;
	const int BN = 128;
	const int BK = 8;
	const int TM = 8;
	const int TN = 8;
	
	__shared__ double a_shared[BM][BK];
	__shared__ double b_shared[BK][BN];
	double sub_c[TM][TN] = { 0.0 };

	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int ty = threadIdx.y;
	const int tx = threadIdx.x;
	const int tid = ty * blockDim.x + tx;

	int a_shared_row = tid >> 1;
	int a_shared_col = (tid % 2) << 2;

	int b_shared_row = tid >> 5;
	int b_shared_col = (tid % 32) << 2;

	int a_global_row = by * BM + a_shared_row;
	int b_global_col = bx * BN + b_shared_col;

	for (int bk = 0; bk < (k + BK - 1) / BK; bk++) {
		int a_global_col = bk * BK + a_shared_col;
		int a_global_site = a_global_row + a_global_col * row;

		int b_global_row = bk * BK + b_shared_row;
		int b_global_site = b_global_row + b_global_col * k;
		for (int i = 0; i < 4; i++) {
			a_shared[a_shared_row][a_shared_col + i] = (a_global_row < row && a_global_col < k) ? a[a_global_site + i * row] : 0;
			b_shared[b_shared_row][b_shared_col + i] = (b_global_row < k && b_global_col < col) ? b[b_global_site + i * k] : 0;
		}
		__syncthreads();
		
		#pragma unroll
		for (int j = 0; j < BK; j++) {
			#pragma unroll
			for (int m = 0; m < TM; m++) {
				int a_shared_inc_row = ty * TM + m;
				#pragma unroll
				for (int n = 0; n < TN; n++) {
					int b_shared_inc_col = tx * TN + n;
					sub_c[m][n] += a_shared[a_shared_inc_row][j] * b_shared[j][b_shared_inc_col];
				}
			}
		}
		__syncthreads();
	}
	#pragma unroll
	for (int m = 0; m < TM; m++) {
		int c_global_row = by * BM + ty * TM + m;
		if (c_global_row >= row) { break; }
		#pragma unroll
		for (int n = 0; n < TN; n += 4) {
			int c_global_col = bx * BN + tx * TN + n;
			if (c_global_col >= col) { break; }
			int c_global_site = c_global_row + c_global_col * row;
			for (int i = 0; i < 4; i++) {
				if (c_global_col + i >= col) { break; }
				c[c_global_site + i * row] = sub_c[m][n + i];
			}
		}
	}
}

cudaError_t matmul_shared(MatrixXd& a, MatrixXd& b, MatrixXd& c) {
	double* aptr;
	double* bptr;
	double* cptr;
	size_t a_size = sizeof(double) * a.size();
	size_t b_size = sizeof(double) * b.size();
	size_t c_size = sizeof(double) * c.size();
	cudaError_t cudaError;
	cudaError = cudaMalloc(&aptr, a_size);
	cudaError = cudaMalloc(&bptr, b_size);
	cudaError = cudaMalloc(&cptr, c_size);
	cudaError = cudaMemcpy(aptr, a.data(), a_size, cudaMemcpyHostToDevice);
	cudaError = cudaMemcpy(bptr, b.data(), b_size, cudaMemcpyHostToDevice);
	dim3 blocksdim(16, 16, 1);
	dim3 gridsdim((b.cols() - 1) / 128 + 1, (a.rows() - 1) / 128 + 1, 1);
	multi_matrix_shared << <gridsdim, blocksdim >> > (aptr, bptr, cptr, a.rows(), b.cols(), a.cols());
	cudaError = cudaGetLastError();
	cudaError = cudaDeviceSynchronize();
	cudaError = cudaMemcpy(c.data(), cptr, c_size, cudaMemcpyDeviceToHost);
	cudaError = cudaFree(aptr);
	cudaError = cudaFree(bptr);
	cudaError = cudaFree(cptr);
	return cudaError;
}

__global__ void add_matrix(double* a, double* b, double* c, int row, int col) {
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < col && y < row) {
		c[y + x * row] = a[y + x * row] + b[y + x * row];
	}
}

cudaError_t matadd(MatrixXd& a, MatrixXd& b, MatrixXd& c) {
	double* aptr;
	double* bptr;
	double* cptr;
	size_t a_size = sizeof(double) * a.size();
	cudaError_t cudaError;
	cudaError = cudaMalloc(&aptr, a_size);
	cudaError = cudaMalloc(&bptr, a_size);
	cudaError = cudaMalloc(&cptr, a_size);
	cudaError = cudaMemcpy(aptr, a.data(), a_size, cudaMemcpyHostToDevice);
	cudaError = cudaMemcpy(bptr, b.data(), a_size, cudaMemcpyHostToDevice);
	dim3 blockdim(32, 32, 1);
	dim3 griddim((a.cols() - 1) / 32 + 1, (a.rows() - 1) / 32 + 1, 1);
	add_matrix << <griddim, blockdim >> > (aptr, bptr, cptr, a.rows(), a.cols());
	cudaError = cudaGetLastError();
	cudaError = cudaDeviceSynchronize();
	cudaError = cudaMemcpy(c.data(), cptr, a_size, cudaMemcpyDeviceToHost);
	cudaError = cudaFree(aptr);
	cudaError = cudaFree(bptr);
	cudaError = cudaFree(cptr);
	return cudaError;
}