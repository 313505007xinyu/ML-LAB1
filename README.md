# ML-LAB1
## 1.做法說明
### Tool : 
1. Colab
2. Python
### library:
1. numpy
2. pandas
3. seaborn
4. matplotlib
### Abstract:
本競賽選擇以Colab作為開發環境來進行房價預測。將資料上傳，計算各個feature和price之間的關聯性，把和price較無關聯的feature刪除，利用關聯性較高的feature創造出更多和price較有關聯的feature，幫助模型進行預測。透過相互比較各個模型的結果，挑選出最效果最好的模型，並透過調整模型參數，來提升準確率。

資料前處理:
1.先檢查缺失值情況print(df.isnull().sum())
檢查結果顯示:
鄉鎮市區缺475筆
都市土地使用分區缺38769筆
非都市土地使用分區缺418015筆
分都市土地使用編定缺418251筆
移轉層次缺171筆
主要用途缺185筆
主要建材缺65筆
建築物完成年月缺455784筆
單價元平方公尺889
車位類別79935
備註412170
建案名稱51126
棟及號127008
解約情形455499

2.將多數都為缺失的值疑除，90%的缺失值比例，並存為trainv2
檢查結果顯示:
鄉鎮市區缺475筆
移轉層次缺171筆
主要用途缺185筆
主要建材缺65筆
單價元平方公尺889


3.將剩餘缺失的值用眾數(最常見的值)來填補
此處不選中位數為填補值的原因是因為資料為中文，不得用中位數
再將結果存成trainv3，處理完的資料檢查結果為沒有缺失值

訓練模型:
1. 導入庫
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
pandas 和 numpy 用於數據操作。
seaborn 和 matplotlib 用於數據可視化。
train_test_split 用於分割數據集。
XGBRegressor 用於建立 XGBoost 回歸模型。
mean_absolute_error, r2_score 用於評估模型性能。
StandardScaler 用於標準化數據。
2. 讀取訓練資料
df_train = pd.read_csv('trainv12.csv')
print("檢查缺失值：")
print(df_train.isnull().sum())
從 'trainv12.csv' 讀取訓練資料集。
檢查各欄位是否有缺失值。
3. 數據清理
df_train.drop(columns=['id', 'sale_yr'], inplace=True)
df_train.fillna(df_train.mean(), inplace=True)
刪除無用的欄位：'id' 和 'sale_yr' 不需要參與模型訓練。
填補缺失值：用各欄位的均值來填補缺失值，這樣可以避免模型中因缺失值造成的錯誤。
4. 處理特殊值
carprice_median = df_train['carprice'][df_train['carprice'] > 0].median()
df_train['carprice'] = df_train['carprice'].replace(0, carprice_median)
car_square_median = df_train['car_square'][df_train['car_square'] > 0].median()
df_train['car_square'] = df_train['car_square'].replace(0, car_square_median)
將 carprice 和 car_square 中的 0 值替換為該欄位的中位數，避免這些欄位對訓練模型產生不良影響。
5. 分割特徵和目標變量
X_train = df_train.drop(columns=['price'])
Y_train = df_train['price']
price 是目標變量（要預測的房價），其餘欄位為特徵。
6. 讀取驗證資料集
df_valid = pd.read_csv('validv4.csv')
df_valid.drop(columns=['sale_yr'], inplace=True)
df_valid.fillna(df_valid.mean(), inplace=True)
df_valid['carprice'] = df_valid['carprice'].replace(0, carprice_median)
df_valid['car_square'] = df_valid['car_square'].replace(0, car_square_median)
X_valid = df_valid.drop(columns=['price'])
Y_valid = df_valid['price']
從 'validv4.csv' 讀取驗證資料，並進行相似的清理操作，包括刪除無用欄位和填補缺失值。
7. 讀取測試資料集
df_test = pd.read_csv('testv5.csv')
id_test = df_test['id']
df_test.drop(columns=['id', 'sale_yr'], inplace=True)
df_test.fillna(df_test.mean(), inplace=True)
df_test['carprice'] = df_test['carprice'].replace(0, carprice_median)
df_test['car_square'] = df_test['car_square'].replace(0, car_square_median)
X_test = df_test
測試資料集：從 'testv5.csv' 讀取並進行清理，保留 id 以便後續輸出結果。
8. 特徵標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
標準化特徵：使用 StandardScaler 對訓練、驗證和測試資料集進行標準化，以使模型能更好地收斂。
9. 建立和訓練模型
model = XGBRegressor(
    learning_rate=0.1,
    n_estimators=200,
    max_depth=5,
    min_child_weight=10,
    random_state=42
)
model.fit(X_train, Y_train)
XGBRegressor 模型：設置一些參數，如學習率、樹的數量、樹的深度等，並用訓練資料集來訓練模型。
10. 評估模型性能
Y_valid_pred = model.predict(X_valid)
mae_valid = mean_absolute_error(Y_valid, Y_valid_pred)
r2_valid = r2_score(Y_valid, Y_valid_pred)
print("驗證集模型性能：")
print(f"MAE (平均絕對誤差): {mae_valid:.3f}")
print(f"R^2 (決定係數): {r2_valid:.3f}")
驗證集評估：使用驗證資料集進行預測，計算 平均絕對誤差（MAE） 和 決定係數（R^2），以衡量模型性能。
11. 預測測試集
Y_pred = model.predict(X_test)
Y_pred = np.maximum(Y_pred, 0)
預測測試集：對測試集進行預測，並將負值替換為 0（房價不應該為負）。
12. 導出結果
result = pd.DataFrame({'id': id_test, 'price': Y_pred})
result.to_csv('predictions.csv', index=False)
輸出預測結果：將測試集的預測房價輸出為 'predictions.csv'，方便後續查看。
總結
這段程式碼從房價數據集出發，進行資料清理和特徵工程，訓練了一個 XGBoost 模型來預測房價，並在驗證資料集上進行了性能評估。該模型的預測結果最終以 CSV 格式保存下來。通過標準化數據和合理的數據清理方法，這份代碼旨在增強模型的泛化能力和準確性。 
