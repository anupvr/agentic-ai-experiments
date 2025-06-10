import os
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from datetime import datetime

# Load the API key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# LLM setup
llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=openai_key)

# Tool 1: Calculator
def calculator(expr: str) -> str:
    try:
        return str(eval(expr))
    except Exception as e:
        return f"Error: {e}"

# Tool 2: Time reporter
def tell_time(_: str) -> str:
    return f"The current time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# Register tools
tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for math like 2+2 or 15*3 or 81/9"
    ),
    Tool(
        name="TimeTool",
        func=tell_time,
        description="Useful for telling the current time."
    )
]

# Agent setup
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Prompt that uses both tools + reasoning
question = "What is 12 times 4, and also tell me what time it is. Then tell me if the product is greater than the number of hours left today."

# Run the agent
response = agent.run(question)
print("\n🤖 Final Answer:", response)