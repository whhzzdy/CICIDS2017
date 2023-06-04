import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.decomposition import PCA

print("1. 数据预处理")
# 加载数据集（请根据实际情况修改文件路径）
data = pd.read_csv("data2.csv")
# 查看原始类别标签的数量
original_class_counts = data[' Label'].value_counts()
print("Original class counts:\n", original_class_counts)

# 输出Heartbleed类的数量
heartbleed_count = original_class_counts['Heartbleed']
print("Number of samples with 'Heartbleed' label:", heartbleed_count)


# 对类别标签进行编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data[' Label'])

# 分离特征和标签
X = data.drop(columns=[' Label'])
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

print("1.5. 降维")
# 保留95%的方差
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# 查看降维后的特征数量
print("Number of features after PCA:", X_pca.shape[1])

print("2. 特征选择")
tree_selector = SelectFromModel(xgb.XGBClassifier())
X_new = tree_selector.fit_transform(X_pca, y)

# 获取选取的特征的列索引
selected_feature_indices = tree_selector.get_support(indices=True)
print("Selected feature indices:", selected_feature_indices)

# 获取PCA降维后的特征名称
pca_feature_names = [f"PCA_{i}" for i in range(X_pca.shape[1])]

# 获取选取的特征的名称
selected_feature_names = [pca_feature_names[i] for i in selected_feature_indices]
print("Selected feature names:", selected_feature_names)

print("3. 数据划分")
X_train, X_test, y_train, y_test = train_test_split(X_new, y_encoded, test_size=0.2, random_state=42)

print("4. 数据平衡")
smote = SMOTE(random_state=42, k_neighbors=2)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("5. 模型训练")
xgb_clf = xgb.XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y_encoded)), n_jobs=-1, scale_pos_weight=1)

# 使用GridSearchCV进行超参数调优
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
}

grid_search = GridSearchCV(xgb_clf, param_grid, scoring='balanced_accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

# 输出最佳参数组合
print("Best parameters found: ", grid_search.best_params_)

# 使用最佳参数重新训练模型
best_xgb_clf = grid_search.best_estimator_

print("6. 模型评估")
# 使用交叉验证评估模型性能
cv_scores = cross_val_score(best_xgb_clf, X_train_resampled, y_train_resampled, cv=5, scoring='balanced_accuracy')
print("Cross-validated Balanced Accuracy: {:.2f}".format(np.mean(cv_scores)))

# 在测试集上评估模型性能
y_pred = best_xgb_clf.predict(X_test)
accuracy = balanced_accuracy_score(y_test, y_pred)
print("Balanced Accuracy on Test Set: {:.2f}".format(accuracy))

# 输出分类报告
report = classification_report(y_test, y_pred, zero_division=1)
print("Classification Report:\n", report)

print("7. 模型应用")
# ...（与之前的代码相同）


1. 数据预处理
Original class counts:
 BENIGN                        2273097
DoS Hulk                       231073
PortScan                       158930
DDoS                           128027
DoS GoldenEye                   10293
FTP-Patator                      7938
SSH-Patator                      5897
DoS slowloris                    5796
DoS Slowhttptest                 5499
Bot                              1966
Web Attack � Brute Force         1507
Web Attack � XSS                  652
Infiltration                       36
Web Attack � Sql Injection         21
Heartbleed                         11
Name:  Label, dtype: int64
Number of samples with 'Heartbleed' label: 11
Class counts:
 0     2273097
4      231073
10     158930
2      128027
3       10293
7        7938
11       5897
6        5796
5        5499
1        1966
12       1507
14        652
9          36
13         21
8          11
dtype: int64
1.5. 降维
Number of features after PCA: 25
2. 特征选择
Selected feature indices: [ 0  1  2  3  4  6 13 17 19]
Selected feature names: ['PCA_0', 'PCA_1', 'PCA_2', 'PCA_3', 'PCA_4', 'PCA_6', 'PCA_13', 'PCA_17', 'PCA_19']
3. 数据划分
4. 数据平衡
5. 模型训练
