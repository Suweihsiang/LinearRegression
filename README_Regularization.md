Regularization.h  Regularization.cpp  
使用C++實現Lasso、LassoLarsIC、Ridge  
===========================================================================  
  
Lasso  
===========================================================================  
Parameters
===========================================================================  
**iters：int**  
迭代次數，預設為1000  
  
**alpha：double**  
L1係數，預設為100  

**error：double**  
當迭代之誤差小於error時，則提前停止訓練  
    
Attributes  
===========================================================================  
**r_2：double**  
訓練後之R2分數  
  
**IC：double**  
AIC或BIC，模型配適統計值  
  
**coef：VectorXd**  
訓練後所獲得之係數向量  
  
**history：vector\<double\>**  
存放每次迭代之方差  

    
Methods  
===========================================================================  
===========================================================================  
**Lasso()**  
Lasso建構式  
  
===========================================================================  
**~Lasso()**  
Lasso解構式  
  
===========================================================================  
**void set_params(unordered_map<string,double> params)**  
設定Lasso演算法之參數(iter、error、alpha)  
**parameters:**  
params：unordered_map<string,double>  
   *{參數名稱：參數設定值}*  
    
===========================================================================  
**void get_params()**  
印出Lasso演算法之參數  
  
===========================================================================  
**void fit(MatrixXd& x, VectorXd& y)**  
進行Lasso演算法之配適  
**parameters:**  
x：MatrixXd  
   *自變數*  
y：VectorXd  
   *應變數*  
  
===========================================================================  
**VectorXd predict(MatrixXd& x, VectorXd& coef)**  
**parameters:**  
x：MatrixXd  
   *自變數*  
coef：VectorXd  
   *各項特徵之係數*  
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
**double calc_IC(MatrixXd& x, VectorXd& y,string criterion = "aic")**  
**parameters:**  
x：MatrixXd  
   *自變數*  
y：VectorXd  
   *應變數*  
criterion：string  
   *模型配適統計值之評估準則，可選AIC及BIC，預設為AIC*  
**return:**  
IC：double  
   *模型配適統計值*  
  
===========================================================================  
**vector\<double\> gethistory()**  
**return:**  
history：vector\<double\>  
   *每次迭代之方差*  
  
===========================================================================  
**void save_result(string path, string mode = "app")**  
儲存Lasso演算法訓練之結果  
**parameters:**  
path：string  
   *儲存檔案之絕對路徑*  
mode：string  
   *儲存方式，new為重新儲存新的結果，app為在舊的結果後接著儲存新的結果*  
  
===========================================================================  
**double calc_noise_var(MatrixXd& x, VectorXd& y)**  
**parameters:**  
x：MatrixXd  
   *自變數*  
y：VectorXd  
   *應變數*  
**return:**  
noise_var：double  
   *模型雜訊之方差*  
  
===========================================================================  
**double get_degree_of_freedom(VectorXd& coef)**  
**parameters:**  
coef：VectorXd  
   *訓練後獲得之特徵係數*  
**return:**  
degree_of_freedom：double  
   *模型自由度*  
  
  
Lasso_LARS  
===========================================================================  
Parameters
===========================================================================  
**iters：int**  
迭代次數，預設為10000  
  
**alpha_min：double**  
可接受之最小L1係數，當L1係數小於此值時停止迭代，以達特徵篩選之目的，預設為0  
  
Attributes  
===========================================================================  
**coef_path：vector\<VectorXd\>**  
訓練後所獲得之特徵係數路徑，與alpha_path相對應  
  
**alpha_path：vector\<double\>**  
訓練後所獲得之L1係數路徑，與coef_path相對應  
  
**criterions：vector\<double\>**  
alpha_path及coef_path對應之模型配適統計值  
  
**noise_variance：double**  
模型雜訊之方差  
  
**IC：double**  
AIC或BIC，所有criterion中最小之模型配適統計值  
  
**alpha：double**  
所獲得之IC其所對應之L1係數  
  
**best_coef：VectorXd**  
所獲得之IC其所對應之特徵係數為最佳係數  
  
Methods  
===========================================================================  
===========================================================================  
**Lasso_LARS()**  
Lasso_LARS建構式  
  
===========================================================================  
**~Lasso_LARS()**  
Lasso_LARS解構式  
  
===========================================================================  
**void set_params(unordered_map<string,double> params)**  
設定演算法之參數(iter、alpha)  
**parameters:**  
params：unordered_map<string,double>  
   *{參數名稱：參數設定值}*  
    
===========================================================================  
**void get_params()**  
印出演算法之參數  
  
===========================================================================  
**void fit(MatrixXd& x, VectorXd& y,string criterion,bool fit_intercept)**  
進行演算法之配適  
**parameters:**  
x：MatrixXd  
   *自變數*  
y：VectorXd  
   *應變數*  
criterion：string  
   *模型配飾統計值適用之準則，可為aic或bic*  
fit_intercept：bool  
   *是否加入常數*  
  
===========================================================================  
**vector\<VectorXd\> get_coef_path()**  
**return:**  
coef_path：vector\<VectorXd\>  
   *訓練後獲得之特徵係數路徑*  
  
===========================================================================  
**vector\<double\> get_alpha_path()**  
**return:**  
alpha_path：vector\<double\>  
   *訓練後獲得之L1係數路徑*  
  
===========================================================================  
**vector\<double\> get_criterions()**  
**return:**  
criterions：vector\<double\>  
   *對應L1係數路徑及特徵係數路徑之模型配適統計值*  
  
===========================================================================  
**double calc_IC(MatrixXd& x, VectorXd& y, VectorXd& coef, string criterion, bool fit_intercept, double noise_var)**  
**parameters:**  
x：MatrixXd  
   *自變數*  
y：VectorXd  
   *應變數*  
coef：VectorXd  
   *最佳特徵係數*  
criterion：string  
   *模型配適統計值之評估準則，可選AIC及BIC，預設為AIC*  
fit_intercept：bool  
   *是否加入常數*  
noise_var：double  
   *模型雜訊變異量*  
**return:**  
IC：double  
   *模型配適統計值*  
  
===========================================================================  
**void save_result(string path)**  
儲存Lasso演算法訓練之結果  
**parameters:**  
path：string  
   *儲存檔案之絕對路徑*  
  
===========================================================================  
**double calc_noise_var(MatrixXd& x, VectorXd& y, bool fit_intercept)**  
**parameters:**  
x：MatrixXd  
   *自變數*  
y：VectorXd  
   *應變數*  
fit_intercept：bool  
   *是否加入常數*  
**return:**  
noise_var：double  
   *模型雜訊之方差*  
  
===========================================================================  
**double get_degree_of_freedom(VectorXd& coef,bool fit_intercept)**  
**parameters:**  
coef：VectorXd  
   *訓練後獲得之特徵係數*  
fit_intercept：bool  
   *是否加入常數*  
**return:**  
degree_of_freedom：double  
   *模型自由度*  
  
  
Ridge  
===========================================================================  
Parameters
===========================================================================  
**alpha：double**  
L2係數，預設為100  
      
Attributes  
===========================================================================  
**r_2：double**  
訓練後之R2分數  
    
**coef：VectorXd**  
訓練後所獲得之係數向量  
  
Methods  
===========================================================================  
===========================================================================  
**Ridge()**  
Ridge建構式  
  
===========================================================================  
**~Ridge()**  
Ridge解構式  
  
===========================================================================  
**void set_params(unordered_map<string,double> params)**  
設定演算法之參數(alpha)  
**parameters:**  
params：unordered_map<string,double>  
   *{參數名稱：參數設定值}*  
    
===========================================================================  
**void get_params()**  
印出演算法之參數設定值  
  
===========================================================================  
**void fit(MatrixXd& x, VectorXd& y)**  
進行演算法之配適  
**parameters:**  
x：MatrixXd  
   *自變數*  
y：VectorXd  
   *應變數*  
    
===========================================================================  
**VectorXd predict(MatrixXd& x, VectorXd& coef)**
**parameters:**  
x：MatrixXd  
   *自變數*  
coef：VectorXd  
   *各項特徵之係數*  
**return:**  
y_pred：VectorXd  
   *預測值*  
  
===========================================================================  
**vector\<VectorXd\> get_coef()**	 
**return:**  
coef：vector\<VectorXd\>  
   *訓練後獲得之特徵係數*  
    
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
**void save_result(string path, string mode = "app")**  
儲存Lasso演算法訓練之結果  
**parameters:**  
path：string  
   *儲存檔案之絕對路徑*  
mode：string  
   *儲存方式，new為重新儲存新的結果，app為在舊的結果後接著儲存新的結果*  