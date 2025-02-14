import json

# JSON源文件名
json_filename = 'data/validation_custom.json'
# 输出的JSONL文件名
jsonl_filename = 'data/output.jsonl'

# 读取JSON文件
with open(json_filename, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 打开JSONL文件准备写入
with open(jsonl_filename, 'w', encoding='utf-8') as jsonl_file:
    # 遍历数据中的每个对话
    for dialogue in data:
        # 将当前对话写入JSONL文件
        jsonl_file.write(json.dumps({"question": dialogue["instruction"], "answer": dialogue["output"]}, ensure_ascii=False) + '\n')
        # 将历史对话写入JSONL文件
        for history_item in dialogue["history"]:
            jsonl_file.write(json.dumps({"question": history_item[0], "answer": history_item[1]}, ensure_ascii=False) + '\n')

print(f"转换完成，结果已保存到 {jsonl_filename}")
