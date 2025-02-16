from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.vllm import Vllm

# 初始化一个HuggingFaceEmbedding对象，将文本向量化
embed_model = HuggingFaceEmbedding(
    model_name="D:/Workspace/llm/model/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

Settings.embed_model = embed_model

llm = HuggingFaceLLM(
    model_name="D:/Workspace/llm/model/Qwen/Qwen1___5-1___8B-Chat",
    tokenizer_name="D:/Workspace/llm/model/Qwen/Qwen1___5-1___8B-Chat",
    model_kwargs={"trust_remote_code": True},
    tokenizer_kwargs={"trust_remote_code": True}
)
# pip install vllm
# CUDA (12.1, 11.8)
# llm = Vllm(
#     model="D:/Workspace/llm/model/Qwen/Qwen1___5-1___8B-Chat",
#     tensor_parallel_size=4,
#     max_new_tokens=100,
#     vllm_kwargs={"swap_space": 1, "gpu_memory_utilization": 0.5},
# )

Settings.llm = llm

# 指定目录读取文档
# documents = SimpleDirectoryReader(input_files=["D:/Workspace/git/ai_llm_base/day31_demo/data/dataset.xlsx"]).load_data()
documents = SimpleDirectoryReader("D:/Workspace/git/ai_llm_base/day31_demo/data").load_data()

# 创建节点解析器
node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)

# 将文档分割成节点
base_node = node_parser.get_nodes_from_documents(documents=documents)

# VectorStoreIndex调用了embed_model，为每个节点的文本创建向量嵌入
# index = VectorStoreIndex.from_documents(documents)
index = VectorStoreIndex(nodes=base_node)

index.storage_context.persist()  # 将索引持久化到本地向量数据库

# as_query_engine引擎调用了llm
query_engine = index.as_query_engine()

# rsp = query_engine.query("教育局10对应的所有的代理机构有哪些")
rsp = query_engine.query("传感器在中国有哪些品牌")

# rsp = llm.chat(messages=[ChatMessage(content="传感器在中国有哪些品牌")])
print(rsp)
