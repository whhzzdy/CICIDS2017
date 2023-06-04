import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. 数据预处理
# 假设你已经将所有CSV文件合并为一个名为"data.csv"的文件
sample_data = pd.read_csv("data2.csv")
print("合并数据集")
# 减少数据集大小
#sample_data = data.sample(frac=0.5, random_state=42)

# 处理缺失值
sample_data = sample_data.fillna(sample_data.mean(numeric_only=True))

# 检查和处理 NaN 值、无穷大值和过大值
sample_data.replace([np.inf, -np.inf], np.finfo(np.float32).max, inplace=True)

# 对标签进行编码
label_encoder = LabelEncoder()
sample_data[" Label"] = label_encoder.fit_transform(sample_data[" Label"])
print("处理数据")
# 将数据集划分为训练集和测试集
X = sample_data.drop(" Label", axis=1)
y = sample_data[" Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 特征选择
# 使用随机森林计算特征重要性
rf = RandomForestClassifier(n_jobs=-1)
rf.fit(X_train, y_train)
importances = rf.feature_importances_

# 根据特征重要性选择前k个重要特征
# 根据特征重要性选择前k个重要特征
k = 10
top_k_indices = np.argsort(importances)[-k:]
top_k_features = X_train.columns[top_k_indices]
print("选取特征:")
for feature, importance in zip(top_k_features, importances[top_k_indices]):
    print(feature, ":", importance)
X_train_selected = X_train.iloc[:, top_k_indices]
X_test_selected = X_test.iloc[:, top_k_indices]

# 3. 模型训练
# 使用随机森林训练模型
clf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1)
clf.fit(X_train_selected, y_train)

# 4. 模型评估
y_pred = clf.predict(X_test_selected)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)

合并数据集
处理数据
选取特征:
Init_Win_bytes_forward : 0.02839997558808195
 Packet Length Mean : 0.03537326627750036
Total Length of Fwd Packets : 0.035864555776639974
 Packet Length Variance : 0.03759756301716146
 Avg Bwd Segment Size : 0.038742342329527245
 Bwd Packet Length Std : 0.03889373166108754
 Destination Port : 0.039136638144921486
 Bwd Packet Length Mean : 0.03998272128116984
 Average Packet Size : 0.05011675066668057
 Packet Length Std : 0.05313231587881751
/environment/miniconda3/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Accuracy: 0.9905696203649569
Precision: 0.9899962960344594
Recall: 0.9905696203649569
F1 Score: 0.9898448011292031
Confusion Matrix:
 [[451436      0      1      4   2684     12      0      0      0      0
     297      0      0      0      0]
 [   258    132      0      0      0      0      0      0      0      0
       0      0      0      0      0]
 [    60      0  25523      0      0      0      0      0      0      0
       0      0      0      0      0]
 [   372      0      0   1701      0      1      0      0      0      0
       0      0      0      0      0]
 [   165      0      0      2  46319      0      0      0      0      0
       0      0      0      0      0]
 [   600      0      0      0      0    468      4      0      0      0
       0      0      0      0      0]
 [   333      0      0      0      0      4    836      0      0      0
       0      0      0      0      0]
 [     3      0      0      0      0      0      0   1609      0      0
       0      0      0      0      0]
 [     0      0      0      0      0      0      0      0      2      0
       0      0      0      0      0]
 [     5      0      0      0      0      0      0      0      0      0
       0      0      0      0      0]
 [    76      0      0      0     13      0      0      0      0      0
   31622      0      0      0      0]
 [    11      0      0      0      0      0      0      0      0      0
       0   1162      0      0      0]
 [   308      0      0      0      0      0      0      0      0      0
       0      0      0      0      0]
 [     6      0      0      0      1      0      0      0      0      0
       0      0      0      0      0]
 [   118      0      0      0      1      0      0      0      0      0
       0      0      0      0      0]]
