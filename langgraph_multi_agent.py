import os
from dotenv import load_dotenv
from datetime import datetime
from langchain.agents import Tool, AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END

# Load API key
load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

# ---------- TOOLS ----------
def calculator(expr: str) -> str:
    try:
        return str(eval(expr))
    except Exception as e:
        return f"Error: {e}"

def current_time(_: str) -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def mock_web_search(query: str) -> str:
    return f"🔎 Web search result for: '{query}' (mocked)"

# ---------- TOOLS WRAPPED FOR AGENTS ----------
calc_tool = Tool(name="Calculator", func=calculator, description="Useful for math problems like 123*4")
time_tool = Tool(name="Clock", func=current_time, description="Returns current date and time")
search_tool = Tool(name="Search", func=mock_web_search, description="Returns search result for query")

# ---------- AGENTS ----------
calculator_agent = initialize_agent([calc_tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
clock_agent = initialize_agent([time_tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
search_agent = initialize_agent([search_tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
fallback_agent = initialize_agent([], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

# ---------- LANGGRAPH NODES ----------

# Intent classifier
def classify_intent(state: dict) -> dict:
    question = state["question"]
    response = llm.invoke(f"What is the intent of this query: '{question}'?\nRespond with one of: 'math', 'time', 'search', or 'unknown'.")
    intent = response.content.strip().lower()
    print(f"🧠 Intent classified as: {intent}")
    return {"intent": intent}

# Agent wrappers
def run_calc(state): return {"response": calculator_agent.run(state["question"])}
def run_clock(state): return {"response": clock_agent.run(state["question"])}
def run_search(state): return {"response": search_agent.run(state["question"])}
def run_fallback(state): return {"response": fallback_agent.run(state["question"])}

# Router
def route_by_intent(state):
    intent = state.get("intent", "")
    if intent in ["math", "time", "search"]:
        return intent
    return "fallback"

# Final node
def final_output(state: dict) -> dict:
    print(f"✅ Final Response:\n{state['response']}")
    return state

# ---------- LANGGRAPH STRUCTURE ----------
builder = StateGraph(dict)
builder.add_node("classify", RunnableLambda(classify_intent))
builder.add_node("math", RunnableLambda(run_calc))
builder.add_node("time", RunnableLambda(run_clock))
builder.add_node("search", RunnableLambda(run_search))
builder.add_node("fallback", RunnableLambda(run_fallback))
builder.add_node("output", RunnableLambda(final_output))

builder.set_entry_point("classify")
builder.add_conditional_edges("classify", route_by_intent, {
    "math": "math",
    "time": "time",
    "search": "search",
    "fallback": "fallback"
})

# Route to final output
builder.add_edge("math", "output")
builder.add_edge("time", "output")
builder.add_edge("search", "output")
builder.add_edge("fallback", "output")
builder.set_finish_point("output", END)

graph = builder.compile()

# ---------- TESTING ----------
if __name__ == "__main__":
    print("\n--- Example 1: Math ---")
    graph.invoke({"question": "What is 45*3?"})

    print("\n--- Example 2: Time ---")
    graph.invoke({"question": "What time is it now?"})

    print("\n--- Example 3: Search ---")
    graph.invoke({"question": "Who won the 2022 World Cup?"})

    print("\n--- Example 4: Unknown Intent (fallback) ---")
    graph.invoke({"question": "Tell me a joke about AI."})