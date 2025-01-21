# pip install -qU langchain-text-splitters

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载示例文档
with open("../../resource/knowledge.txt", encoding="utf-8") as f:
    knowledge = f.read()
text_splitter = RecursiveCharacterTextSplitter(
    # 设置一个非常小的块大小，只是为了展示。
    chunk_size=100,
    # 块之间的目标重叠。重叠的块有助于在上下文分割时减少信息丢失。
    chunk_overlap=20,
    # 确定块大小的函数。
    length_function=len,
    # 分隔符列表（默认为 ["\n\n", "\n", " ", ""]）是否应被解释为正则表达式。
    is_separator_regex=False,
)
texts = text_splitter.create_documents([knowledge])
print(texts[0])
print(texts[1])

# chunk_size设为100,chunk_overlap设为20
# chunk 1: 第1-100个token
# chunk 2: 第81-180个token (与chunk 1重叠20个)
# page_content='﻿I am honored to be with you today at your commencement from one of the finest universities in the'
# page_content='universities in the world. I never graduated from college. Truth be told, this is the closest I've'

print(text_splitter.split_text(knowledge)[:2])
