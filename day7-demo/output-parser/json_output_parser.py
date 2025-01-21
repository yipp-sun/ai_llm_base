# pip install -qU langchain langchain-openai
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
#langchain v0.2 使用以下引用pydantic
# from langchain_core.pydantic_v1 import BaseModel, Field
#langchain v0.3 使用以下引用pydantic
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o", temperature=0)

# 定义您期望的数据结构。相当于langchain帮我加了一些提示词
class Joke(BaseModel):
    setup: str = Field(description="设置笑话的问题")
    punchline: str = Field(description="解决笑话的答案")


# 还有一个用于提示语言模型填充数据结构的查询意图。
joke_query = "告诉我一个笑话。"
# 设置解析器 + 将指令注入提示模板。
parser = JsonOutputParser(pydantic_object=Joke)
prompt = PromptTemplate(
    template="回答用户的查询。\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
print(parser.get_format_instructions())
chain = prompt | model | parser
response = chain.invoke({"query": joke_query})
print(response)
"""
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"properties": {"setup": {"description": "设置笑话的问题", "title": "Setup", "type": "string"}, "punchline": {"description": "解决笑话的答案", "title": "Punchline", "type": "string"}}, "required": ["setup", "punchline"]}
```
{'setup': '为什么计算机不能在海里游泳？', 'punchline': '因为它们害怕网络崩溃！'}
"""
