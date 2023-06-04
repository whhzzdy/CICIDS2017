import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.decomposition import PCA
import time

print("加载数据")
#加载数据集（请根据实际情况修改文件路径）
data = pd.read_csv("data2.csv")

# 将数据集分为正常流量和异常流量
normal_data = data[data[' Label'] == 'BENIGN']
dos_hulk_data = data[data[' Label'] == 'DoS Hulk']
portscan_data = data[data[' Label'] == 'PortScan']
ddos_data = data[data[' Label'] == 'DDoS']

# 保存原始数据为CSV文件
#normal_data.to_csv('BENIGN_original.csv', index=False)
#dos_hulk_data.to_csv('DoS_Hulk_original.csv', index=False)
#portscan_data.to_csv('PortScan_original.csv', index=False)
#ddos_data.to_csv('DDoS_original.csv', index=False)

print("提取5w数据")
# 分别取5万条数据作为新的数据集
sample_size = 50000
normal_data_sample = normal_data.sample(sample_size)
dos_hulk_data_sample = dos_hulk_data.sample(sample_size)
portscan_data_sample = portscan_data.sample(sample_size)
ddos_data_sample = ddos_data.sample(sample_size)
# 将抽取的数据合并为一个新的数据集
sample_data = pd.concat([normal_data_sample, dos_hulk_data_sample, portscan_data_sample, ddos_data_sample], ignore_index=True)


print("1. 数据预处理")
# 查看原始类别标签的数量
original_class_counts = sample_data[' Label'].value_counts()
print("Original class counts:\n", original_class_counts)
# 对类别标签进行编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(sample_data[' Label'])

# 分离特征和标签
X = sample_data.drop(columns=[' Label'])
y = pd.Series(y_encoded)
# 查看每个类别的样本数量
class_counts = y.value_counts()
print("Class counts:\n", class_counts)

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


print("3. 数据划分")
X_train, X_test, y_train, y_test = train_test_split(X_scaled ,y_encoded, test_size=0.2, random_state=42)

print("4. 模型训练")
xgb_clf = xgb.XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y_encoded)), n_jobs=-1)
xgb_clf.fit(X_train, y_train)

from sklearn.metrics import precision_score, recall_score, f1_score

print("5. 模型评估")
# 使用交叉验证评估模型性能
start_time = time.time()
cv_scores = cross_val_score(xgb_clf, X_train, y_train, cv=5, scoring='balanced_accuracy')
print("Cross-validated Balanced Accuracy: {:.2f}".format(np.mean(cv_scores)))
print("Time taken for cross-validation: {:.2f} seconds".format(time.time() - start_time))

# 在测试集上评估模型性能
start_time = time.time()
y_pred = xgb_clf.predict(X_test)
accuracy = balanced_accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("Balanced Accuracy on Test Set: {:.2f}".format(accuracy))
print("Precision on Test Set: {:.2f}".format(precision))
print("Recall on Test Set: {:.2f}".format(recall))
print("F1 Score on Test Set: {:.2f}".format(f1))
print("Time taken for test set evaluation: {:.2f} seconds".format(time.time() - start_time))

# 输出分类报告
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=1)
print("Classification Report:\n", report)

加载数据
提取5w数据
1. 数据预处理
Original class counts:
 BENIGN      50000
DoS Hulk    50000
PortScan    50000
DDoS        50000
Name:  Label, dtype: int64
Class counts:
 0    50000
2    50000
3    50000
1    50000
dtype: int64
3. 数据划分
4. 模型训练
5. 模型评估
Cross-validated Balanced Accuracy: 1.00
Time taken for cross-validation: 202.27 seconds
Balanced Accuracy on Test Set: 1.00
Precision on Test Set: 1.00
Recall on Test Set: 1.00
F1 Score on Test Set: 1.00
Time taken for test set evaluation: 0.12 seconds
Classification Report:
               precision    recall  f1-score   support

      BENIGN       1.00      1.00      1.00     10001
        DDoS       1.00      1.00      1.00     10091
    DoS Hulk       1.00      1.00      1.00      9992
    PortScan       1.00      1.00      1.00      9916

    accuracy                           1.00     40000
   macro avg       1.00      1.00      1.00     40000
weighted avg       1.00      1.00      1.00     40000
