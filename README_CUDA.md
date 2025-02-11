utils.cuh  utils.cu  
使用CUDA實現矩陣運算  
===========================================================================  
Functions  
===========================================================================  
===========================================================================  
**void Check_cuda_Error(cudaError_t error, const char * const file, const int line)**  
在進行記憶體配置時，確認是否有錯誤之情況發生，若錯誤情形發生時，打印出相關訊息  
**parameters:**  
error：cudaError_t  
   *CUDA錯誤訊息*  
file：const char* const  
   *發生錯誤之檔案名稱*  
line：const int  
   *錯誤發生之行數*  
    
===========================================================================  
**void multi_matrix(double * as, double * bs, double * cs, int row, int col, int k)**  
提供matmul函式記憶體配置後進行矩陣乘法  
**parameters:**  
as：double*  
   *矩陣a之記憶體位置*  
bs：double*  
   *矩陣b之記憶體位置*  
cs：double*  
   *將運算結果儲存至矩陣c，該矩陣之記憶體位置*  
row：int  
   *矩陣a之列數*  
col：int  
   *矩陣b之行數*  
k：int  
   *矩陣a之行數(矩陣b之列數)*  
  
===========================================================================  
**void matmul(MatrixXd& a, T& b, T& c)**  
矩陣乘法  
**parameters:**  
a：MatrixXd  
   *尺寸為(row,k)*  
b：T  
   *尺寸為(k,col)，T可以為MatrixXd、VectorXd*  
c：T  
   *矩陣a和矩陣b計算結果，尺寸為(row,col)，T可以為MatrixXd、VectorXd*    
  
===========================================================================  
**void multi_matrix_shared(double * a,double * b, double * c, int row, int col, int k)**  
提供matmul_shared函式記憶體配置後進行矩陣乘法  
**parameters:**  
as：double*  
   *矩陣a之記憶體位置*  
bs：double*  
   *矩陣b之記憶體位置*  
cs：double*  
   *將運算結果儲存至矩陣c，該矩陣之記憶體位置*  
row：int  
   *矩陣a之列數*  
col：int  
   *矩陣b之行數*  
k：int  
   *矩陣a之行數(矩陣b之列數)*  
  
===========================================================================  
**void matmul_shared(MatrixXd& a, T& b, T& c)**  
矩陣乘法(使用shared memory)  
**parameters:**  
a：MatrixXd  
   *尺寸為(row,k)*  
b：T  
   *尺寸為(k,col)，T可以為MatrixXd、VectorXd*  
c：T  
   *矩陣a和矩陣b計算結果，尺寸為(row,col)，T可以為MatrixXd、VectorXd*  
  
===========================================================================  
**void multi_matvec_shared(double * a, double * b, double * c, int row, int k)**  
提供matvecmul_shared函式記憶體配置後進行矩陣與向量之乘法  
**parameters:**  
a：double*  
   *矩陣a之記憶體位置*  
b：double*  
   *向量b之記憶體位置*  
c：double*  
   *將運算結果儲存至向量c，該向量之記憶體位置*  
row：int  
   *矩陣a之列數*   
k：int  
   *矩陣a之行數(向量b之列數)*  
  
===========================================================================  
**void matvecmul_shared(MatrixXd& a, VectorXd& b, VectorXd& c)**  
矩陣與向量之乘法(使用shared memory)  
**parameters:**  
a：MatrixXd  
   *尺寸為(row,k)*  
b：VectorXd  
   *尺寸為(k,1)*  
c：VectorXd  
   *矩陣a和矩陣b計算結果，尺寸為(row,1)*  
    
===========================================================================  
**void add_matrix(double * a, double * b, double * c, int row, int col)**  
提供matadd函式記憶體配置後進行矩陣加法  
**parameters:**  
a：double*  
   *矩陣a之記憶體位置*  
b：double*  
   *矩陣b之記憶體位置*  
c：double*  
   *將運算結果儲存至矩陣c，該矩陣之記憶體位置*  
row：int  
   *矩陣a與矩陣b之列數*   
col：int  
   *矩陣a與矩陣b之行數*  
  
===========================================================================  
**void matadd(MatrixXd& a, MatrixXd& b, MatrixXd& c)**  
矩陣加法  
**parameters:**  
a：MatrixXd  
   *尺寸為(row,col)*  
b：MatrixXd  
   *尺寸為(row,col)*  
c：MatrixXd  
   *矩陣a和矩陣b計算結果，尺寸為(row,col)*