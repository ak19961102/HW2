## 導入lightgbm
import lightgbm as lgb
## 導入Scikit-Learn的評量套件
from sklearn import metrics
from sklearn.metrics import mean_squared_error
## 導入Scikit-Learn的內建數據集
from sklearn. datasets import load_iris
## 導入Scikit-Learn用來拆分訓練集和測試集的套件
from sklearn.model_selection import train_test_split


## 載入數據集
iris_dataset = load_iris()
data = iris_dataset.data
target = iris_dataset.target
## 拆分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split (data, target, test_size = 0.3)

## 創建成符合lgb特徵的數據集格式
## 將數據保存成LightGBM二進位文件，加載速度更快，占用更少內存空間
## 訓練集
lgb_train = lgb.Dataset(X_train, y_train)

## 測試集
lgb_test = lgb.Dataset(X_test, y_test, reference = lgb_train)

## 撰寫訓練用的參數
params = {
  'task': 'train',
  ## 算法類型
  'boosting': 'gbdt',
  'num_trees': 100,
  'num_leaves': 20,
  'max_depth': 6,
  'learning_rate': 0.04,
  ## 構建樹時的特徵選擇比例
  'feature_fraction': 0.5,
  'feature_fraction_seed': 8,
  "bagging_fraction":0.5,
  ## k 表示每k次迭代就進行bagging
  'bagging_freq':5,
  ## 如果數據集樣本分布不均衡，可以幫助明顯提高準確率
  'is_unbalance': True,
  'verbose':0,
  ## 目標函數
  'objective': 'regression',
  ## 度量指標
  'metric': {'rmse', 'auc'},
  # 度量輸出的頻率
  'metric_freq': 1,
}

## 訓練模型
test_results = {}
lgbm = lgb.train(params, lgb_train, valid_sets = lgb_test, num_boost_round = 100)

## 保存模型
lgbm.save_model('save_model.txt')

## 預測測試集
## 在訓練期間有啟動early_stopping_rounds， 就可以透過best_iteration來從最佳送代中獲得預測結果
y_pred = lgbm.predict(X_test, num_iteration = lgbm.best_iteration)
print(y_pred)

## 評估模型的好壞
## RMSE
rmse = mean_squared_error (y_test, y_pred) ** 0.5
print('RMSE of the model: ', rmse)