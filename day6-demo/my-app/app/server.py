from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langserve import add_routes
from langchain.prompts import ChatPromptTemplate

app = FastAPI(
    title="My App LangChain 服务器",
    version="1.0",
    description="使用 Langchain 的 Runnable 接口的简单 API 服务器",
)

add_routes(
    app,
    ChatOpenAI(model="gpt-3.5-turbo"),
    path="/openai",
)

prompt = ChatPromptTemplate.from_template("告诉我一个关于 {topic} 的笑话")
add_routes(
    app,
    prompt | ChatOpenAI(model="gpt-4"),
    path="/openai_ext",
)

# @app.get("/")
# async def redirect_root_to_docs():
#     return RedirectResponse("/docs")


# Edit this to add the chain you want to add
# add_routes(app, NotImplemented)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
