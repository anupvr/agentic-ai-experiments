import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

# Import tools from previously defined RAG agents
from multi_rag_agents import tools

# Load environment variables
load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- State Classifier ----------------
def classify_source(state: dict) -> dict:
    query = state["question"]
    response = llm.invoke(f"Based on the question below, which source is most relevant: 'txt', 'pdf', 'docx', or 'web'?\n\n{query}")
    source = response.content.strip().lower()
    print(f"🧠 Classifier selected: {source}")
    return {"source": source}

# ---------------- Agent Runners ----------------
def run_txt(state): return {"response": tools["txt"].run(state["question"])}
def run_pdf(state): return {"response": tools["pdf"].run(state["question"])}
def run_docx(state): return {"response": tools["docx"].run(state["question"])}
def run_web(state): return {"response": tools["web"].run(state["question"])}

# ---------------- Router ----------------
def route_by_source(state):
    src = state.get("source", "web")
    return src if src in tools else "web"

# ---------------- Final Output ----------------
def final_output(state: dict) -> dict:
    print("\n📦 Final State:")
    print(state)
    print(f"\n✅ Answer:\n{state.get('response')}")
    return state

# ---------------- LangGraph Build ----------------
builder = StateGraph(dict)
builder.set_entry_point("classify")

# Add nodes
builder.add_node("classify", RunnableLambda(classify_source))
builder.add_node("txt", RunnableLambda(run_txt))
builder.add_node("pdf", RunnableLambda(run_pdf))
builder.add_node("docx", RunnableLambda(run_docx))
builder.add_node("web", RunnableLambda(run_web))
builder.add_node("output", RunnableLambda(final_output))

# Routing
builder.add_conditional_edges("classify", route_by_source, {
    "txt": "txt",
    "pdf": "pdf",
    "docx": "docx",
    "web": "web"
})

# Edges to output
builder.add_edge("txt", "output")
builder.add_edge("pdf", "output")
builder.add_edge("docx", "output")
builder.add_edge("web", "output")
builder.set_finish_point("output", END)

graph = builder.compile()

# ---------------- Run Example ----------------
if __name__ == "__main__":
    print("\n--- Ask Anything ---")
    question = "What year did Einstein publish the general theory of relativity?"
    graph.invoke({"question": question})