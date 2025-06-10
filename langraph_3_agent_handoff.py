
import os
from dotenv import load_dotenv
from langchain.agents import Tool, AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END

# Load API key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_key)

# ---------------- SIMULATED TOOLS ----------------
def planner_tool(task: str) -> str:
    return f"Plan: Break the task '{task}' into [Research topic, Summarize, Present]"

def researcher_tool(topic: str) -> str:
    return f"Research Notes: '{topic}' includes facts A, B, and C."

def presenter_tool(notes: str) -> str:
    return f"Presentation: Here's a simple explanation of {notes.lower()}."

# ---------------- AGENTS ----------------
def create_agent(name, tool_func, input_key, output_key):
    def runner(state):
        input_data = state.get(input_key, state.get("question"))
        output = tool_func(input_data)
        print(f"🤖 {name} output:", output)
        return {output_key: output}
    return RunnableLambda(runner)

planner_agent = create_agent("Planner", planner_tool, "question", "plan")
researcher_agent = create_agent("Researcher", researcher_tool, "plan", "research_notes")
presenter_agent = create_agent("Presenter", presenter_tool, "research_notes", "final_presentation")

# ---------------- FINAL OUTPUT ----------------
def final_output(state: dict) -> dict:
    print("\n📦 Final Shared State:")
    for k, v in state.items():
        print(f"{k}: {v}")
    print(f"\n✅ Final Presentation:\n{state['final_presentation']}")
    return state

# ---------------- BUILD LANGGRAPH ----------------
builder = StateGraph(dict)
builder.set_entry_point("planner")

builder.add_node("planner", planner_agent)
builder.add_node("researcher", researcher_agent)
builder.add_node("presenter", presenter_agent)
builder.add_node("output", RunnableLambda(final_output))

builder.add_edge("planner", "researcher")
builder.add_edge("researcher", "presenter")
builder.add_edge("presenter", "output")
builder.set_finish_point("output", END)

graph = builder.compile()

# ---------------- RUN ----------------
if __name__ == "__main__":
    print("\n--- 3-Agent Handoff Simulation ---")
    graph.invoke({"question": "Explain how solar panels work"})
