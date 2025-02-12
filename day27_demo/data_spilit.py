# 按照8：2划分train和test

import csv
import json
from random import shuffle


# 定义一个函数，用于将CSV文件分割成训练和测试JSON文件
def split_csv_to_json(csv_file_path, train_ratio=0.8):
    # 创建一个字典，用于春初每个对话ID以及对应的输入输出对
    conversions = {}
    with open(csv_file_path, mode="r", encoding="utf-8") as file:
        # 创建一个DictReader对象，用于读取CSV文件中的每一行
        reader = csv.DictReader(file)
        # 遍历CSV文件中的每一行
        for row in reader:
            # 获取当前行的对话ID
            conversion_id = row["Conversation ID"]
            # 如果对话ID不在字典中，则初始化一个空列表
            if conversion_id not in conversions:
                conversions[conversion_id] = []
            # 将当前行的输入和输出添加到对应对话ID的列表中
            conversions[conversion_id].append({
                "input": row["Input"],
                "output": row["Output"]
            })

    # 定义一个内部函数，将对话列表转换为JSON格式
    def conversions_to_json(conversation_list):
        # 使用列表推导式，将每个对话列表转换为包含单个字典的列表
        return [{"conversation": conv} for conv in conversation_list]

    # 获取所有对话ID的列表，并随机打乱顺序
    conversion_ids = list(conversions.keys())
    shuffle(conversion_ids)
    # 计算训练集和测试集的分割点
    split_index = int(len(conversion_ids) * train_ratio)

    # 根据分割点获取训练集和测试集的对话数据
    train_conversions = [conversions[conv_id] for conv_id in conversion_ids[:split_index]]
    test_conversions = [conversions[conv_id] for conv_id in conversion_ids[split_index:]]

    # 将训练集和测试集的对话数据转换为JSON格式
    train_json = conversions_to_json(train_conversions)
    test_json = conversions_to_json(test_conversions)

    # 将训练数据写入train_data.json文件
    with open("data/train_data.json", "w", encoding="utf-8") as file:
        json.dump(train_json, file, ensure_ascii=False, indent=4)

    # 将测试数据写入test_data.json文件
    with open("data/test_data.json", "w", encoding="utf-8") as file:
        json.dump(test_json, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    csv_file_path = "data/output_conversations.csv"
    split_csv_to_json(csv_file_path)
