from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Tool (FIXED parentheses)
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [wiki]

# Prompt
prompt = ChatPromptTemplate.from_template(
    "Answer the question using tools if necessary.\nQuestion: {input}\n{agent_scratchpad}"
)

# Agent (correct function)
agent = create_tool_calling_agent(llm, tools, prompt)

# Executor (REQUIRED)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run
response = agent_executor.invoke({"input": "What is reinforcement learning?"})

print(response["output"])
