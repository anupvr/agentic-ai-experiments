# pip install langchain langchain-openai langchain-experimental python-dotenv
import os
from dotenv import load_dotenv
from datetime import datetime
from langchain.tools import tool
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_experimental.plan_and_execute import PlanAndExecute
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor
from langchain_experimental.plan_and_execute.planners.chat_planner import load_chat_planner


# Load .env and keys
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Setup LLM
llm = ChatOpenAI(temperature=0.3, model="gpt-4o", api_key=openai_key)

# Memory for context tracking
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Tools (internal or dummy for this context)

def get_current_time(_: str) -> str:
    return f"The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."


def reflect_on_learning(_: str) -> str:
    return (
        "This week I explored the architecture of agentic AI systems. "
        "I learned how to connect tools, memory, and planning to simulate intelligent behavior. "
        "Next, I plan to dive into LangGraph for complex workflows!"
    )

tools = [
    Tool(name="TimeTool", func=get_current_time, description="Returns the current time."),
    Tool(name="LearningReflectionTool", func=reflect_on_learning, description="Returns a summary of key learnings.")
]

# Plan and execute agent
#planner = LLMChainPlanner(llm=llm)
#executor = LLMSingleActionAgentExecutor.from_llm_and_tools(llm=llm, tools=tools, verbose=True)
#llm = ChatOpenAI(model="gpt-4", temperature=0)

# Load the planner using the provided function
planner = load_chat_planner(llm)

# Load the executor with your tools (assuming you have defined them)
executor = load_agent_executor(llm=llm, tools=tools)

agent = PlanAndExecute(
    planner=planner,
    executor=executor,
    memory=memory,
    verbose=True
)

# Run task
if __name__ == "__main__":
    task = "Write a simple LinkedIn post about what I learned this week in agentic AI."
    print("\n🔹 Task:")
    print(task)
    result = agent.invoke({"input": task})
    print("\n📝 LinkedIn Post:")
    print(result["output"])
