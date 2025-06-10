#pip install langgraph
import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_key)

# Define state type (dictionary-based)
class GraphState(dict):
    question: str
    intent: str
    result: str

# Node 1: Classify user intent (math or time)
def classify_intent(state: dict) -> dict:
    question = state.get("question", "")
    response = llm.invoke(f"Classify the intent of this input as either 'math' or 'time':\n{question}")
    intent = "math" if "math" in response.content.lower() else "time"
    print(f"🧠 Intent classified as: {intent}")
    return {"intent": intent}

# Node 2a: Calculator tool
def calculator_tool(state: dict) -> dict:
    try:
        expr = state["question"]
        result = str(eval(expr))
        return {"result": f"🧮 Answer: {result}"}
    except Exception as e:
        return {"result": f"❌ Calculation failed: {str(e)}"}

# Node 2b: Clock tool
def clock_tool(state: dict) -> dict:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"result": f"⏰ Current time is: {now}"}

# Routing logic
def route_by_intent(state: dict) -> str:
    return "calculator" if state.get("intent") == "math" else "clock"

# Final node (output state)
def final_output(state: dict) -> dict:
    print(f"✅ Final Answer: {state.get('result')}")
    return state

# Build LangGraph
builder = StateGraph(GraphState)
builder.add_node("intent_classifier", RunnableLambda(classify_intent))
builder.add_node("calculator", RunnableLambda(calculator_tool))
builder.add_node("clock", RunnableLambda(clock_tool))
builder.add_node("final_output", RunnableLambda(final_output))

# Define entry and flow
builder.set_entry_point("intent_classifier")
builder.add_conditional_edges("intent_classifier", route_by_intent, {
    "calculator": "calculator",
    "clock": "clock"
})
builder.add_edge("calculator", "final_output")
builder.add_edge("clock", "final_output")
builder.set_finish_point("final_output", END)

# Compile graph
graph = builder.compile()

# Example runs
if __name__ == "__main__":
    print("\n--- Example 1: Math ---")
    graph.invoke({"question": "100 + 23"})

    print("\n--- Example 2: Time ---")
    graph.invoke({"question": "What is the current time?"})