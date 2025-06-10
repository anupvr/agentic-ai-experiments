import os
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
#from langfuse import Langfuse
from langchain.callbacks import LangChainTracer
from langchain.memory import ConversationBufferMemory
import time

# Load env vars
load_dotenv()
#os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
#os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
#os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")

# LangFuse setup
# langfuse = Langfuse()
# trace = langfuse.trace(
#     name="resilient_calculator_agent",
#     user_id="anup_user_01",
#     session_id="session_day9_retries"
# )
#tracer = LangChainTracer(langfuse)

# LLM setup
llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

# Custom tool with error + retry handling
def safe_calculator(expression: str) -> str:
    if "/0" in expression.replace(" ", ""):
        return "🚫 Guardrail triggered: Division by zero is not allowed."
    retries = 2
    for attempt in range(retries + 1):
        try:
            # Simulate error for divide-by-zero
            result = eval(expression)
            return str(result)
        except Exception as e:
            if attempt < retries:
                print(f"⚠️ Calculator failed. Retrying... Attempt {attempt + 1}")
                time.sleep(1)
            else:
                print("❌ Calculator failed after retries. Falling back to LLM.")
                return f"Error: {e}. Please avoid invalid math like divide by zero."

# Tool definitions
tools = [
    Tool(
        name="ResilientCalculator",
        func=safe_calculator,
        description="Evaluates math expressions like 100+20 or 120/0"
    )
]

# Memory (optional but useful)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Agent setup
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    #callbacks=[tracer],
    verbose=True
)

# Run agent with a failing and a normal case
print("\n🧪 Case 1: Valid expression")
print(agent.run("What is 120 divided by 6?"))

print("\n🧪 Case 2: Invalid expression with fallback")
print(agent.run("What is 100 divided by 0?"))