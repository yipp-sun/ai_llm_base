from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.vllm import Vllm
from llama_index.core import StorageContext, load_index_from_storage
import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import ServiceContext

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
# 删除tensor_parallel_size=4，意思是把现在的模型放到4个GPU上面
#     # tensor_parallel_size=4,
#     max_new_tokens=100,
#     vllm_kwargs={"swap_space": 1, "gpu_memory_utilization": 0.5},
# )

Settings.llm = llm

# 定义向量存储数据库
chroma_client = chromadb.PersistentClient()

# 获取集合
chroma_collection = chroma_client.get_or_create_collection("quickstart")

documents = SimpleDirectoryReader("D:/Workspace/git/ai_llm_base/day31_demo/data_md").load_data()
# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# save to disk
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)
# # load from disk
# index = VectorStoreIndex.from_vector_store(
#     vector_store,
#     embed_model=embed_model,
# )

query_engine = index.as_query_engine()

rsp = query_engine.query("OpenCompass是什么？")

# rsp = llm.chat(messages=[ChatMessage(content="传感器在中国有哪些品牌")])
print(rsp)

# # 判断本地数据库是否存在，目的：不用每次都是embedding了
# if not os.path.exists("storage"):
#     documents = SimpleDirectoryReader("D:/Workspace/git/ai_llm_base/day31_demo/data_md").load_data()
#     node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)
#     base_node = node_parser.get_nodes_from_documents(documents=documents)
#     index = VectorStoreIndex.from_documents(documents)
#     # 存储索引
#     index.storage_context.persist()
#     print("未发现本地数据集，开始创建新的数据库并存index！")
# else:
#     print("加载已存在的index数据库！！")
#     # 加载现有索引
#     storage_context = StorageContext.from_defaults(persist_dir="storage")
#     # 从存储中加载索引
#     index = load_index_from_storage(storage_context=storage_context)
# query_engine = index.as_query_engine()
#
# rsp = query_engine.query("OpenCompass是什么？")
#
# # rsp = llm.chat(messages=[ChatMessage(content="传感器在中国有哪些品牌")])
# print(rsp)

# # 指定目录读取文档
# # documents = SimpleDirectoryReader(input_files=["D:/Workspace/git/ai_llm_base/day31_demo/data/dataset.xlsx"]).load_data()
# documents = SimpleDirectoryReader("D:/Workspace/git/ai_llm_base/day31_demo/data_csv").load_data()
# print(documents)
# # 创建节点解析器
# # node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)
#
# # 将文档分割成节点
# # base_node = node_parser.get_nodes_from_documents(documents=documents)
#
# # VectorStoreIndex调用了embed_model，为每个节点的文本创建向量嵌入
# index = VectorStoreIndex.from_documents(documents)
# # index = VectorStoreIndex(nodes=base_node)
#
# index.storage_context.persist()  # 将索引持久化到本地向量数据库
#
# # as_query_engine引擎调用了llm
# query_engine = index.as_query_engine()
#
# # rsp = query_engine.query("教育局10对应的所有的代理机构有哪些")
# # rsp = query_engine.query("传感器在中国有哪些品牌")
# rsp = query_engine.query("项目6的来源是什么？")
#
# # rsp = llm.chat(messages=[ChatMessage(content="传感器在中国有哪些品牌")])
# print(rsp)
