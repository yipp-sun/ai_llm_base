import pandas as pd
import json
from sklearn.model_selection import train_test_split

# 加载数据集
file_path = 'data/output_conversations.csv'  # 替换为你的CSV文件的实际路径
dataset = pd.read_csv(file_path)

# 按照80%训练集和20%验证集的比例分割数据集
train_set, validation_set = train_test_split(dataset, test_size=0.2, random_state=42)


# 定义一个函数，将数据集转换为指定的JSON格式
def convert_to_custom_format(df):
    # 按照对话ID分组对话
    conversations = df.groupby('Conversation ID')
    custom_data = []  # 用于存储转换后的数据

    # 遍历每个对话组
    for conversation_id, group in conversations:
        # 初始化对话字典，按照指定的格式填充字段
        dialogue = {
            "instruction": group.iloc[0]['Input'],  # 假设每个对话组的第一个输入为instruction
            "input": "",  # input字段为空，根据你的要求
            "output": group.iloc[-1]['Output'],  # 假设每个对话组的最后一个输出为output
            "history": []  # 初始化历史对话列表
        }

        # 遍历对话组中的每一行，填充历史对话列表
        for index, row in group.iterrows():
            dialogue['history'].append([row['Input'], row['Output']])

        # 将填充好的对话字典添加到自定义数据列表中
        custom_data.append(dialogue)

    # 返回转换后的数据列表
    return custom_data


# 将训练集和验证集转换为自定义格式
train_custom = convert_to_custom_format(train_set)
validation_custom = convert_to_custom_format(validation_set)

# 定义保存文件的路径
train_file_path_custom = 'data/train_custom.json'  # 替换为你希望保存训练集的路径
validation_file_path_custom = 'data/validation_custom.json'  # 替换为你希望保存验证集的路径

# 将转换后的训练集数据保存为JSON文件
with open(train_file_path_custom, 'w', encoding='utf-8') as f:
    json.dump(train_custom, f, ensure_ascii=False, indent=4)  # 确保输出是UTF-8编码，并格式化JSON

# 将转换后的验证集数据保存为JSON文件
with open(validation_file_path_custom, 'w', encoding='utf-8') as f:
    json.dump(validation_custom, f, ensure_ascii=False, indent=4)  # 确保输出是UTF-8编码，并格式化JSON
