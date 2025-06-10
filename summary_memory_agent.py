import os
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from datetime import datetime

# Load environment
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# LLM setup
llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=openai_key)

# Tools
def calculator(expr: str) -> str:
    try:
        return str(eval(expr))
    except Exception as e:
        return f"Error: {e}"

def tell_time(_: str) -> str:
    now = datetime.now()
    return f"The current time is {now.strftime('%Y-%m-%d %H:%M:%S')} (Hour: {now.hour})"

tools = [
    Tool(name="Calculator", func=calculator, description="Useful for math expressions like 1200/12 or 100+500"),
    Tool(name="TimeTool", func=tell_time, description="Tells current time with hour")
]

# Memory: summary-based
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
    verbose=True
)

# Helper to print summary
def print_summary(stage: str):
    print(f"\n🧾 Summary Snapshot ({stage}):")
    print(memory.buffer)

# Run interactions
print_summary("Before Step 1")
print("\n🔹 Step 1:")
print(agent.run("What is 1200 divided by 12?"))
print_summary("After Step 1")

print("\n🔹 Step 2:")
print(agent.run("Now add that result to 300."))
print_summary("After Step 2")

print("\n🔹 Step 3:")
print(agent.run("What time is it?"))
print_summary("After Step 3")

print("\n🔹 Step 4:")
print(agent.run("Is the final total greater than the current hour?"))
print_summary("After Step 4")