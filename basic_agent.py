import os
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
#from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
# Load the API key from .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Setup the LLM
#llm = OpenAI(temperature=0, model_name="gpt-4o", openai_api_key=openai_key)
llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=openai_key)
# Define a simple math tool
def calculator(expr: str) -> str:
    try:
        return str(eval(expr))
    except Exception as e:
        return f"Error: {e}"

tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for simple math like 2+2 or 21/7+3"
    )
]

# Initialize the agent with tools
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent
question = "What's 15 divided by 3, and then tell me if that number is odd or even?"
#"the result of sin 90 is odd or even"
#"What is 21 divided by 7 plus 3?"
response = agent.run(question)

print("\n🤖 Final Answer:", response)