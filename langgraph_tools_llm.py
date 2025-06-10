import os
import datetime
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Step 1: Define Tools
def calculator_tool(query: str) -> str:
    try:
        return str(eval(query))
    except:
        return "Error: Could not calculate."

def date_tool(_: str) -> str:
    return f"Today's date is: {datetime.date.today()}"

tools = [
    Tool.from_function(
        func=calculator_tool,
        name="Calculator",
        description="Useful for solving math problems"
    ),
    Tool.from_function(
        func=date_tool,
        name="DateTool",
        description="Provides the current date"
    )
]

# Step 2: Setup LLM Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=True
)

# Step 3: Define State class
class AgentState(dict):
    pass

# Step 4: Define Nodes
def router_node(state: AgentState):
    print("Router received state:", state)
    question = state["question"].lower()
    if any(op in question for op in ["+", "-", "*", "/", "square root"]):
        return "calculator"
    elif "date" in question or "day" in question:
        return "date"
    else:
        return "llm_agent"

def calculator_node(state: AgentState):
    return {"result": calculator_tool(state["question"])}

def date_node(state: AgentState):
    return {"result": date_tool(state["question"])}

def llm_agent_node(state: AgentState):
    return {"result": agent.run(state["question"])}

# Step 5: Build LangGraph
graph = StateGraph(AgentState)

graph.add_node("router", router_node)
graph.add_node("calculator", calculator_node)
graph.add_node("date", date_node)
graph.add_node("llm_agent", llm_agent_node)

graph.add_conditional_edges(
    "router",
    router_node,
    {
        "calculator": "calculator",
        "date": "date",
        "llm_agent": "llm_agent"
    }
)

graph.add_edge("calculator", END)
graph.add_edge("date", END)
graph.add_edge("llm_agent", END)

graph.set_entry_point("router")

app = graph.compile()
# Try a math query
res = app.invoke({"question":"What is 15 * 7 - 5?"})
print("Response:", res)

# Ask for the date
res = app.invoke({"question":"What day is it today?"})
print("Response:", res)

# General question
res = app.invoke({"question":"Explain quantum computing in one line."})
print("Response:", res)