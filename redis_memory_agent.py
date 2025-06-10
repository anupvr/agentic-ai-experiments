import os
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain_openai import ChatOpenAI
from datetime import datetime

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Setup LLM
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

# External Redis-based chat memory
chat_history = RedisChatMessageHistory(
    session_id="agentic-ai-session-01",  # Can be per user/session
    url="redis://localhost:6379"         # Default Redis connection
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=chat_history,
    return_messages=True
)

# Agent setup
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Memory viewer
def print_memory_state(stage: str):
    print(f"\n🧾 Redis Memory Snapshot ({stage}):")
    if not memory.chat_memory.messages:
        print("  [No messages yet]")
    else:
        for msg in memory.chat_memory.messages:
            role = "User" if msg.type == "human" else "Agent"
            print(f"{role}: {msg.content}")

# Run interactions
print_memory_state("Before Step 1")
print("\n🔹 Step 1:")
print(agent.run("What is 150 times 2?"))
print_memory_state("After Step 1")

print("\n🔹 Step 2:")
print(agent.run("Now add that result to 100."))
print_memory_state("After Step 2")

print("\n🔹 Step 3:")
print(agent.run("What time is it?"))
print_memory_state("After Step 3")

print("\n🔹 Step 4:")
print(agent.run("Is the final total greater than the current hour?"))
print_memory_state("After Step 4")