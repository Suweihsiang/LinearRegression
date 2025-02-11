LinearRegression.h  LinearRegression.cpp  
使用C++實現LinearRegression  
===========================================================================  
Parameters
===========================================================================  
**iters：int**  
梯度下降法迭代次數，預設為1000  
  
**lr：double**  
梯度下降法學習率，預設為0.01  
  
**error：double**  
當迭代之誤差小於error時，則提前停止訓練  
  
Attributes
===========================================================================  
**r_2：double**  
訓練後之R2分數  
  
**coef：VectorXd**  
訓練後所獲得之係數向量  
  
**history：vector< double >**  
存放每次迭代之方差  
  
Methods  
===========================================================================  
===========================================================================  
**LinearRegression()**  
LinearRegression建構式  
  
===========================================================================  
**~LinearRegression()**  
LinearRegression解構式  
  
===========================================================================  
**void set_params(unordered_map<string,double> params)**  
設定線性回歸之參數  
**parameters:**  
params：unordered_map<string,double>  
   *{參數名稱：參數設定值}，設定iters、lr及error*  
    
===========================================================================  
**void get_params()**  
印出線性回歸之參數  
  
===========================================================================  
**void fit_gd(MatrixXd& x, VectorXd& y, bool CUDA_use, bool fit_intercept)**  
以梯度下降法之方式進行線性回歸  
**parameters:**  
x：MatrixXd  
   *自變數*  
y：VectorXd  
   *應變數*  
CUDA_use：bool  
   *是否使用CUDA*  
fit_intercept：bool  
   *是否加入常數*  
  
===========================================================================  
**void fit_closed_form(MatrixXd& x, VectorXd& y, bool CUDA_use,bool fit_intercept)**  
以closed form之方式獲得線性回歸之係數  
**parameters:**  
x：MatrixXd  
   *自變數*  
y：VectorXd  
   *應變數*  
CUDA_use：bool  
   *是否使用CUDA*  
fit_intercept：bool  
   *是否加入常數*  
  
===========================================================================  
**VectorXd predict(MatrixXd& x, VectorXd& coef, bool CUDA_use)**  
**parameters:**  
x：MatrixXd  
   *自變數*  
coef：VectorXd  
   *各項特徵之係數*  
CUDA_use：bool  
   *是否使用CUDA*  
**return:**  
y_pred：VectorXd  
   *預測值*  
  
===========================================================================  
**VectorXd getCoef()**  
**return:**  
coef：VectorXd  
   *訓練後獲得之係數*  
  
===========================================================================  
**double score(MatrixXd& x, VectorXd& y)**  
**parameters:**  
x：MatrixXd  
   *自變數*  
y：VectorXd  
   *應變數*  
**return:**  
score：double  
   *模型之R2分數*  
  
===========================================================================  
**vector\<double\> gethistory()**  
**return:**  
history：vector\<double\>  
   *每次迭代之方差*  