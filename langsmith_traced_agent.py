# This needs you to have langfuse account and key
#Also needs pip install langchain langchain-openai langfuse openai python-dotenv

import os
from datetime import datetime
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
#from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.chat_models import ChatOpenAI



# Load from .env
load_dotenv()
# Force into os.environ for LangFuse



import os
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.chat_models import ChatOpenAI

# Step 1: Set LangSmith environment variables
os.environ["LANGCHAIN_API_KEY"]= os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]= os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]= "true"

# Step 2: Set your OpenAI key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

print(os.environ["OPENAI_API_KEY"])
# Step 3: Initialize the LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Step 4: Load tools
tools = load_tools(["llm-math"], llm=llm)

# Step 5: Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Step 6: Run a query with optional metadata for LangSmith
result = agent.run(
    input="What is the square root of 2025 plus 23?",
    config={"metadata": {"user_id": "anup_vr", "session": "demo_run"}}
)

# Step 7: Output the result
print("\n✅ Final Answer:", result)