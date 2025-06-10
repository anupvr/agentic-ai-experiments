#pip install langchain langchain-openai langchain-experimental python-dotenv
import os
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    LLMChainPlanner,
    LLMSingleActionAgentExecutor,
)
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langfuse import Langfuse
from langchain.callbacks import LangChainTracer

# Load keys
load_dotenv()

# Forward env values to LangFuse (required for it to pick them up)
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")

# LangFuse tracer setup
langfuse = Langfuse()
tracer = LangChainTracer(langfuse)

# LLM setup
openai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.3, model="gpt-4o", api_key=openai_key)

# Tool to simulate external explanation
def agentic_ai_info(_: str) -> str:
    return (
        "Agentic AI refers to AI systems that don't just respond to commands, "
        "but can plan, reason, and act with some independence toward goals—like an intelligent assistant "
        "that can figure out what steps are needed to help you."
    )

tools = [
    Tool(
        name="ExplainAgenticAI",
        func=agentic_ai_info,
        description="Gives a basic explanation of agentic AI"
    )
]

# Memory (optional but useful)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Planner
planner = LLMChainPlanner(llm=llm, callbacks=[tracer])

# Executor
executor = LLMSingleActionAgentExecutor.from_llm_and_tools(
    llm=llm,
    tools=tools,
    callbacks=[tracer],
    verbose=True
)

# Plan-and-Execute Agent
agent = PlanAndExecute(
    planner=planner,
    executor=executor,
    memory=memory,
    callbacks=[tracer],
    verbose=True
)

# Run the task
if __name__ == "__main__":
    task = "Explain what agentic AI is in a way a 10-year-old can understand."
    print("\n🔹 Task:")
    print(task)
    result = agent.invoke({"input": task})
    print("\n🧠 Final Output:")
    print(result["output"])