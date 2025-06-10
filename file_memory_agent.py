import os
import json
from datetime import datetime
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
from langchain.memory.chat_memory import BaseChatMessageHistory
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# ===== File-backed Chat Memory =====
class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, file_path="chat_memory.json"):
        self.file_path = file_path
        self.messages = self._load_messages()

    def _load_messages(self):
        if not os.path.exists(self.file_path):
            return []
        with open(self.file_path, "r") as f:
            data = json.load(f)
            return [
                HumanMessage(content=m["content"]) if m["type"] == "human" else AIMessage(content=m["content"])
                for m in data
            ]

    def _save_messages(self):
        with open(self.file_path, "w") as f:
            json.dump([{"type": m.type, "content": m.content} for m in self.messages], f)

    def add_message(self, message):
        self.messages.append(message)
        self._save_messages()

    def clear(self):
        self.messages = []
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

# ===== Tools =====
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

# ===== File-based memory setup =====
chat_history = FileChatMessageHistory(file_path="chat_memory.json")
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=chat_history, return_messages=True)

# ===== LLM + Agent =====
llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=openai_key)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# ===== Memory Viewer =====
def print_file_memory(stage: str):
    print(f"\n🧾 File Memory Snapshot ({stage}):")
    if not memory.chat_memory.messages:
        print("  [No messages yet]")
    else:
        for msg in memory.chat_memory.messages:
            role = "User" if msg.type == "human" else "Agent"
            print(f"{role}: {msg.content}")

# ===== Interactions =====
print_file_memory("Before Step 1")
print("\n🔹 Step 1:")
print(agent.run("What is 50 times 3?"))
print_file_memory("After Step 1")

print("\n🔹 Step 2:")
print(agent.run("Add 25 to that result."))
print_file_memory("After Step 2")

print("\n🔹 Step 3:")
print(agent.run("What time is it now?"))
print_file_memory("After Step 3")

print("\n🔹 Step 4:")
print(agent.run("Is the result greater than the current hour?"))
print_file_memory("After Step 4")