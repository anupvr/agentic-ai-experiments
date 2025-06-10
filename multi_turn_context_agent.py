# This needs you to have langfuse account and key
#Also needs pip install langchain langchain-openai langfuse openai python-dotenv
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langchain.callbacks import LangChainTracer

# Load environment
load_dotenv()

# Ensure LangFuse picks up from .env
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")

# LangFuse setup
langfuse = Langfuse()
tracer = LangChainTracer(langfuse)

# OpenAI setup
openai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=openai_key)

# Tools
def calculator(expr: str) -> str:
    try:
        return str(eval(expr))
    except Exception as e:
        return f"Error: {e}"

def get_current_hour(_: str) -> str:
    return str(datetime.now().hour)

tools = [
    Tool(name="Calculator", func=calculator, description="Performs math like 10*5 or 120+40"),
    Tool(name="TimeTool", func=get_current_hour, description="Returns the current hour as a number")
]

# Memory
memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history"
)

# Agent setup
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    callbacks=[tracer],
    verbose=True
)

# Interaction sequence
def run_conversation():
    print(agent.run("What is 20 times 5?"))
    print(agent.run("Add that result to 150."))
    print(agent.run("What is the current hour?"))
    print(agent.run("Now divide the total by the number of hours left today."))

if __name__ == "__main__":
    run_conversation()