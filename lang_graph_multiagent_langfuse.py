import os
from dotenv import load_dotenv
from datetime import datetime
from langchain.agents import Tool, AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langfuse import Langfuse
from langchain.callbacks import LangChainTracer
import time

# Load environment
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Setup LangFuse
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")

langfuse = Langfuse()
trace = langfuse.trace(
    name="multi_agent_langgraph",
    user_id="anup_user",
    session_id="langgraph_retry_session"
)
tracer = LangChainTracer(langfuse)

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_key)

# ---------------- TOOLS ----------------
def calculator(expr: str) -> str:
    try:
        return str(eval(expr))
    except Exception as e:
        raise ValueError(f"Error in calculator: {e}")

def current_time(_: str) -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def mock_web_search(query: str) -> str:
    if "fail" in query.lower():
        raise ValueError("Simulated web search failure.")
    return f"🔎 Mocked search result for '{query}'"

# ---------------- TOOLS AS WRAPPED AGENTS ----------------
def create_agent_with_retry(tools, name, retries=2):
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        callbacks=[tracer]
    )
    
    def agent_runner(state):
        for attempt in range(retries):
            try:
                print(f"🔄 [{name}] Attempt {attempt + 1}")
                response = agent.run(state["question"])
                return {"response": response}
            except Exception as e:
                print(f"⚠️ [{name}] Error: {e}")
                time.sleep(1)
        return {"response": f"❌ [{name}] All retries failed. Please try again."}

    return RunnableLambda(agent_runner)

# Agent nodes with retry logic
calc_node = create_agent_with_retry([Tool("Calculator", calculator, "Arithmetic")], "CalculatorAgent")
clock_node = create_agent_with_retry([Tool("Clock", current_time, "Time check")], "ClockAgent")
search_node = create_agent_with_retry([Tool("Search", mock_web_search, "Mock search")], "SearchAgent")
fallback_node = create_agent_with_retry([], "FallbackAgent")

# ---------------- LANGGRAPH NODES ----------------
def classify_intent(state: dict) -> dict:
    question = state["question"]
    response = llm.invoke(f"What is the intent of this query: '{question}'? Respond with 'math', 'time', 'search', or 'unknown'.")
    intent = response.content.strip().lower()
    print(f"🧠 Intent classified as: {intent}")
    return {"intent": intent}

def route_by_intent(state):
    intent = state.get("intent", "fallback")
    return intent if intent in ["math", "time", "search"] else "fallback"

def final_output(state: dict) -> dict:
    print("\n📦 Final State:")
    print(state)
    print(f"\n✅ Final Response:\n{state['response']}")
    return state

# ---------------- BUILD LANGGRAPH ----------------
builder = StateGraph(dict)
builder.set_entry_point("classify")

builder.add_node("classify", RunnableLambda(classify_intent))
builder.add_node("math", calc_node)
builder.add_node("time", clock_node)
builder.add_node("search", search_node)
builder.add_node("fallback", fallback_node)
builder.add_node("output", RunnableLambda(final_output))

builder.add_conditional_edges("classify", route_by_intent, {
    "math": "math",
    "time": "time",
    "search": "search",
    "fallback": "fallback"
})

builder.add_edge("math", "output")
builder.add_edge("time", "output")
builder.add_edge("search", "output")
builder.add_edge("fallback", "output")
builder.set_finish_point("output", END)

graph = builder.compile()

# ---------------- EXECUTION ----------------
if __name__ == "__main__":
    print("\n--- Example 1: Math ---")
    graph.invoke({"question": "15 * 2"}, config={"callbacks": [tracer]})

    print("\n--- Example 2: Time ---")
    graph.invoke({"question": "What time is it?"}, config={"callbacks": [tracer]})

    print("\n--- Example 3: Web Search ---")
    graph.invoke({"question": "Search for Python vs Java"}, config={"callbacks": [tracer]})

    print("\n--- Example 4: Simulated Failure ---")
    graph.invoke({"question": "fail this search"}, config={"callbacks": [tracer]})

    print("\n--- Example 5: Unknown Intent (Fallback) ---")
    graph.invoke({"question": "Sing a song about data science"}, config={"callbacks": [tracer]})
