# pip install pypdf

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

file_path = ("../../resource/pytorch.pdf")
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()
# "text-embedding-ada-002"这个嵌入模型是为了将文本（如单词、短语或整段文本）转换为数值形式的向量，使得计算机能够处理和理解自然语言。
# faiss是存在内存中的
faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
# k <-- top key
docs = faiss_index.similarity_search("What is PyTorch?", k=2)
for doc in docs:
    print(str(doc.metadata["page"]) + ":", doc.page_content[:300])
