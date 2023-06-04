import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.decomposition import PCA

print("Loading dataset...")
# 加载原始数据集（请根据实际情况修改文件路径）
data = pd.read_csv("data2.csv")

print("Performing undersampling...")
# 下采样
normal_data = data[data[' Label'] == 'BENIGN']
anomaly_data = data[data[' Label'] != 'BENIGN']
sampled_normal_data = normal_data.sample(n=len(anomaly_data), random_state=42)
balanced_data = pd.concat([sampled_normal_data, anomaly_data], axis=0)

print("Encoding labels...")
# 对类别标签进行编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(balanced_data[' Label'])

print("Splitting features and labels...")
# 分离特征和标签
X = balanced_data.drop(columns=[' Label'])
y = pd.Series(y_encoded)

print("Preprocessing data...")
# 找出包含无穷大或非常大数值的列
infinite_columns = X.columns[np.isinf(X).any()]

# 将这些值替换为列的中位数
for col in infinite_columns:
    X[col].replace([np.inf, -np.inf], np.nan, inplace=True)
    X[col].fillna(X[col].median(), inplace=True)

# 填充其他缺失值
X.fillna(X.mean(), inplace=True)

# 将负值替换为非负值
for col in X.columns:
    if X[col].min() < 0:
        X[col] = X[col] - X[col].min()

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Performing PCA...")
# PCA降维
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print("Splitting dataset into train and test sets...")
# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42)

print("Training model...")
# 模型训练
xgb_clf = xgb.XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y_encoded)), n_jobs=-1)
xgb_clf.fit(X_train, y_train)

print("Evaluating model...")
# 模型评估
y_pred = xgb_clf.predict(X_test)
accuracy = balanced_accuracy_score(y_test, y_pred)
print("Balanced Accuracy on Test Set: {:.2f}".format(accuracy))

# 输出分类报告
report = classification_report(y_test, y_pred, zero_division=1)
print("Classification Report:\n", report)

Loading dataset...
Performing undersampling...
Encoding labels...
Splitting features and labels...
Preprocessing data...
Performing PCA...
Splitting dataset into train and test sets...
Training model...
Evaluating model...
Balanced Accuracy on Test Set: 0.84
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00    111320
           1       0.89      0.84      0.86       405
           2       1.00      1.00      1.00     25639
           3       0.99      0.99      0.99      2024
           4       1.00      1.00      1.00     46447
           5       1.00      0.99      0.99      1092
           6       1.00      0.99      1.00      1179
           7       1.00      1.00      1.00      1554
           8       1.00      0.75      0.86         4
           9       1.00      0.50      0.67         8
          10       1.00      1.00      1.00     31717
          11       0.99      0.99      0.99      1220
          12       0.71      0.85      0.77       308
          13       1.00      0.40      0.57         5
          14       0.52      0.24      0.33       137

    accuracy                           1.00    223059
   macro avg       0.94      0.84      0.87    223059
weighted avg       1.00      1.00      1.00    223059
