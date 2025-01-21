from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType

db = SQLDatabase.from_uri("sqlite:///langchain.db")
toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0))
# 工具包能够取得建表语句
print(toolkit.get_tools())
# Agent的强大在于，可以分析出toolkit，把toolkit的SQL变成提示词，并通过大模型处理后，自然语句输出
# Agent是一个指挥系统，调用大模型，然后执行整合，不是大模型，而是个指挥别人干活，然后自己汇总结果做反馈的
# 如果要了解Agent，需要使用LangSmith，因为Agent的链路都比较深
agent_executor = create_sql_agent(
    llm=ChatOpenAI(temperature=0, model="gpt-4"),
    toolkit=toolkit,
    verbose=False,
    agent_type=AgentType.OPENAI_FUNCTIONS
)
result = agent_executor.invoke("Describe the full_llm_cache table")
print(result)
"""
[QuerySQLDatabaseTool(description="Input to this tool is a detailed and correct SQL query, output is a result from the database. If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again. If you encounter an issue with Unknown column 'xxxx' in 'field list', use sql_db_schema to query the correct table fields.", db=<langchain_community.utilities.sql_database.SQLDatabase object at 0x000001E8A5DB1AD0>), InfoSQLDatabaseTool(description='Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables. Be sure that the tables actually exist by calling sql_db_list_tables first! Example Input: table1, table2, table3', db=<langchain_community.utilities.sql_database.SQLDatabase object at 0x000001E8A5DB1AD0>), ListSQLDatabaseTool(db=<langchain_community.utilities.sql_database.SQLDatabase object at 0x000001E8A5DB1AD0>), QuerySQLCheckerTool(description='Use this tool to double check if your query is correct before executing it. Always use this tool before executing a query with sql_db_query!', db=<langchain_community.utilities.sql_database.SQLDatabase object at 0x000001E8A5DB1AD0>, llm=ChatOpenAI(client=<openai.resources.chat.completions.Completions object at 0x000001E8A69ED910>, async_client=<openai.resources.chat.completions.AsyncCompletions object at 0x000001E8A450B9D0>, root_client=<openai.OpenAI object at 0x000001E8A5DF0D10>, root_async_client=<openai.AsyncOpenAI object at 0x000001E8A683E110>, temperature=0.0, model_kwargs={}, openai_api_key=SecretStr('**********')), llm_chain=LLMChain(verbose=False, prompt=PromptTemplate(input_variables=['dialect', 'query'], input_types={}, partial_variables={}, template='\n{query}\nDouble check the {dialect} query above for common mistakes, including:\n- Using NOT IN with NULL values\n- Using UNION when UNION ALL should have been used\n- Using BETWEEN for exclusive ranges\n- Data type mismatch in predicates\n- Properly quoting identifiers\n- Using the correct number of arguments for functions\n- Casting to the correct data type\n- Using the proper columns for joins\n\nIf there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.\n\nOutput the final SQL query only.\n\nSQL Query: '), llm=ChatOpenAI(client=<openai.resources.chat.completions.Completions object at 0x000001E8A69ED910>, async_client=<openai.resources.chat.completions.AsyncCompletions object at 0x000001E8A450B9D0>, root_client=<openai.OpenAI object at 0x000001E8A5DF0D10>, root_async_client=<openai.AsyncOpenAI object at 0x000001E8A683E110>, temperature=0.0, model_kwargs={}, openai_api_key=SecretStr('**********')), output_parser=StrOutputParser(), llm_kwargs={}))]
{'input': 'Describe the full_llm_cache table', 'output': 'The `full_llm_cache` table has the following structure:\n\nColumn | Type | Key\n--- | --- | ---\nprompt | VARCHAR | NOT NULL\nllm | VARCHAR | NOT NULL\nidx | INTEGER | NOT NULL\nresponse | VARCHAR |\n\nPrimary Key: prompt, llm, idx\n\nAnd here\'s some sample data from the `full_llm_cache` table:\n\nprompt: `[{"lc": 1, "type": "constructor", "id": ["langchain", "schema", "messages", "HumanMessage"], "kwargs`\n\nllm: `{"id": ["langchain", "chat_models", "openai", "ChatOpenAI"], "kwargs": {"max_retries": 2, "model_nam`\n\nidx: `0`\n\nresponse: `{"lc": 1, "type": "constructor", "id": ["langchain", "schema", "output", "ChatGeneration"], "kwargs`'}
"""