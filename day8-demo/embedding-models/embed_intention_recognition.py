from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# 数据导入，RAG
loader = TextLoader(".././resource/qa.txt", encoding="UTF-8")
docs = loader.load()
# 数据切分
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
# 创建embedding
embeddings = OpenAIEmbeddings()
# 通过向量数据库存储
vector = FAISS.from_documents(documents, embeddings)
# 查询检索
# 创建 prompt
prompt = ChatPromptTemplate.from_template("""仅根据提供的上下文回答以下问题：:
<context>
{context}
</context>
Question: {input}""")
# 创建模型
llm = ChatOpenAI()
# 创建 document 的chain 查询，做大模型提示词加载的chain
document_chain = create_stuff_documents_chain(llm, prompt)
# RAG的chain，做向量检索用的
# RAG就是给大模型灌一些文档，然后让他回答跟这个文档相关的东西，做一些总结归纳
from langchain.chains import create_retrieval_chain

# 创建搜索chain 返回值为 VectorStoreRetriever，拿到向量数据库检索的对象，通过这个接口，可以使用任意向量数据库，做向量匹配，
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
# 执行请求，意图匹配
# response = retrieval_chain.invoke({"input": "你能帮我做什么事情"})
response = retrieval_chain.invoke({"input": "音乐"})
print(response["answer"])

# 我可以帮您订餐、查询天气、播放音乐、设置闹钟或讲笑话。
