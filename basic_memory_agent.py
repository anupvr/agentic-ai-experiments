import os
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from datetime import datetime

# Load API key
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
    return f"The current time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

tools = [
    Tool(name="Calculator", func=calculator, description="Useful for math expressions."),
    Tool(name="TimeTool", func=tell_time, description="Tells current time.")
]

# Add memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize agent with memory
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Run a series of interactions
print(agent.run("What is 8 times 5?"))

print(agent.run("Also, what time is it now?"))
print(agent.run("Can you remind me what the result was for the first question?"))
