from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,Settings
from llama_index.core.node_parser import SimpleNodeParser

#初始化一个HuggingFaceEmbedding对象，将文本向量化
embed_model = HuggingFaceEmbedding(
    model_name="/root/app/llm/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/Ceceliachenen/paraphrase-multilingual-MiniLM-L12-v2"
)

Settings.embed_model = embed_model

llm = HuggingFaceLLM(
    model_name="/root/app/llm/Qwen/Qwen2___5-3B-Instruct_cusm",
    tokenizer_name="/root/app/llm/Qwen/Qwen2___5-3B-Instruct_cusm",
    model_kwargs={"trust_remote_code":True},
    tokenizer_kwargs={"trust_remote_code":True}
)

Settings.llm = llm

#指定目录读取文档
documents = SimpleDirectoryReader("/root/app/project/demo_20241219/data").load_data()

#创建节点解析器
node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)

#将文档分割成节点
base_node = node_parser.get_nodes_from_documents(documents=documents)



# index = VectorStoreIndex.from_documents(documents)
index = VectorStoreIndex(nodes=base_node)

# index.storage_context.persist()#将索引持久化到本地向量数据库

query_engine = index.as_query_engine()

rsp = query_engine.query("传感器在中国有哪些品牌")

# rsp = llm.chat(messages=[ChatMessage(content="传感器在中国有哪些品牌")])
print(rsp)
