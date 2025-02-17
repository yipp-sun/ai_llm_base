from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.llms import ChatMessage

llm = HuggingFaceLLM(
    # model_name="/root/app/llm/Qwen/Qwen2___5-0___5B-Instruct",
    model_name="D:/Workspace/llm/model/Qwen/Qwen1___5-1___8B-Chat",
    tokenizer_name="D:/Workspace/llm/model/Qwen/Qwen1___5-1___8B-Chat",
    model_kwargs={"trust_remote_code": True},
    tokenizer_kwargs={"trust_remote_code": True}
)

rsp = llm.chat(messages=[ChatMessage(content="xtuner是什么？")])
print(rsp)
# print(llm)会报递归错误
# print(llm)
