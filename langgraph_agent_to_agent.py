
import os
from dotenv import load_dotenv
from datetime import datetime
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
import time

# Load environment
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_key)
shared_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ---------------- TOOLS ----------------
def calculator(expr: str) -> str:
    try:
        return str(eval(expr))
    except Exception as e:
        raise ValueError(f"Error: {e}")

def current_time(_: str) -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------------- AGENTS ----------------
def create_agent_runner(name, tools, custom_prompt_func=None):
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        memory=shared_memory,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )

    def runner(state):
        question = state["question"]
        if custom_prompt_func:
            question = custom_prompt_func(state)
        print(f"🔁 [{name}] Final prompt: {question}")
        try:
            response = agent.run(question)
            return {f"{name.lower()}_response": response}
        except Exception as e:
            return {f"{name.lower()}_response": f"❌ Failed: {e}"}

    return RunnableLambda(runner)

# Agents
calc_agent = create_agent_runner("Calculator", [Tool("Calculator", calculator, "Do math")])
clock_agent = create_agent_runner("Clock", [Tool("Clock", current_time, "Get time")], 
    custom_prompt_func=lambda state: (
        f"The previous answer was: {state.get('calculator_response', 'unknown')}."
        f"Now, what is the current time?"
    )
)

# ---------------- GRAPH ----------------
def final_output(state: dict) -> dict:
    print("\n📦 Final Shared State:")
    for k, v in state.items():
        print(f"{k}: {v}")
    print(f"\n✅ Combined Output:\n{state.get('clock_response') or state.get('calculator_response')}")
    return state

builder = StateGraph(dict)
builder.set_entry_point("calculator")

builder.add_node("calculator", calc_agent)
builder.add_node("clock", clock_agent)
builder.add_node("output", RunnableLambda(final_output))

builder.add_edge("calculator", "clock")
builder.add_edge("clock", "output")
builder.set_finish_point("output", END)

graph = builder.compile()

# ---------------- RUN ----------------
if __name__ == "__main__":
    print("\n--- Agent-to-Agent Communication ---")
    graph.invoke({"question": "What is 40 + 2?"})
