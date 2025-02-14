import pandas as pd
import json

# 加载Excel文件
file_path = 'data/问题答案样例数据.xlsx'  # 请替换为你的Excel文件路径
df = pd.read_excel(file_path)

# 创建一个列表来保存转换后的数据
transformed_data = []

# 遍历DataFrame的每一行
for index, row in df.iterrows():
    # 为每一行创建一个字典，字典具有所需的格式
    data_dict = {
        "instruction": row["问"],
        "input": "",
        "output": row["答"]
    }
    # 将字典添加到列表中
    transformed_data.append(data_dict)

# 将列表转换为JSON格式的字符串
json_data = json.dumps(transformed_data, ensure_ascii=False, indent=4)

# 将JSON数据保存到文件中
json_file_path = 'data/转换后的数据.json'  # 请替换为你希望保存的JSON文件路径
with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json_file.write(json_data)
