import pandas as pd
import os

# 列出包含CSV文件的目录
csv_directory = "DATA"

# 获取目录中的所有CSV文件名
csv_files = [f for f in os.listdir(csv_directory) if f.endswith(".csv")]

# 读取并合并所有CSV文件
merged_data = pd.DataFrame()
for csv_file in csv_files:
    file_path = os.path.join(csv_directory, csv_file)
    data = pd.read_csv(file_path)
    merged_data = pd.concat([merged_data, data], ignore_index=True)

# 将合并后的数据保存为一个新的CSV文件
merged_data.to_csv("data2.csv", index=False)
