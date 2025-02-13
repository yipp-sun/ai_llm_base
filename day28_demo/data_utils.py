#将数据转换为llama factory格式
import pandas as pd
from sklearn.model_selection import train_test_split
import json

# 加载数据集
filepath = 'data/output_conversations.csv'  # 替换为你的CSV文件的实际路径
dataset = pd.read_csv(filepath)

# 按照80%训练集和20%验证集的比例分割数据集
# 确保同一个Conversation ID的所有对话都在同一个集合中
conversation_ids = dataset['Conversation ID'].unique()
train_ids, validation_ids = train_test_split(conversation_ids, test_size=0.2, random_state=42)

train_set = dataset[dataset['Conversation ID'].isin(train_ids)].reset_index(drop=True)
validation_set = dataset[dataset['Conversation ID'].isin(validation_ids)].reset_index(drop=True)

# 定义一个函数来转换数据集为所需的JSON格式
def convert_to_custom_format(df):
    conversations = []

    # 按Conversation ID分组
    for conversation_id, group in df.groupby('Conversation ID'):
        # 获取最后一轮对话
        last_turn = group.iloc[-1]
        instruction = last_turn['Input']
        output = last_turn['Output']

        # 获取除了最后一轮对话之外的所有对话
        history = group.iloc[:-1][['Input', 'Output']].values.tolist()

        conversation = {
            "instruction": instruction,
            "input": "",
            "output": output,
            "system": "你叫小聚，你是由Aron团队在2024年开发的在线心里咨询AI专家。",
            "history": history
        }
        conversations.append(conversation)

    return conversations

# 将训练集和验证集转换为自定义格式
train_custom = convert_to_custom_format(train_set)
validation_custom = convert_to_custom_format(validation_set)

# 保存转换后的训练集数据为JSON文件
train_filepath_custom = 'data/train_custom.json'
with open(train_filepath_custom, 'w', encoding='utf-8') as f:
    json.dump(train_custom, f, ensure_ascii=False, indent=4)  # 确保输出是UTF-8编码，并格式化JSON

# 将转换后的验证集数据保存为JSON文件
validation_filepath_custom = 'data/validation_custom.json'
with open(validation_filepath_custom, 'w', encoding='utf-8') as f:
    json.dump(validation_custom, f, ensure_ascii=False, indent=4)  # 确保输出是UTF-8编码，并格式化JSON
