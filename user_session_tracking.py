from langfuse import Langfuse
from langchain.callbacks import LangChainTracer

# Setup LangFuse
langfuse = Langfuse()
tracer = LangChainTracer(langfuse)

# Start a trace manually (optional but useful)
trace = langfuse.trace(
    name="agentic_ai_langgraph_flow",
    user_id="anup_user_01",           # <- your identifier
    session_id="session_agentic_001", # <- your session or workflow run
    metadata={"origin": "langgraph-demo"}
)

# Attach to LangChain / LangGraph nodes
callbacks = [tracer]