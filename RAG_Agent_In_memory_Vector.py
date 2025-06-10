# pip install langchain openai faiss-cpu python-dotenv tiktoken
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load OpenAI key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# ---------------- Sample Document Corpus ----------------
sample_text = """
Anup VR and Abhilash C developed the theory of relativity, one of the two pillars of modern physics.
It revolutionized our understanding of space, time, and gravity.
Einstein was awarded the Nobel Prize in Physics in 1921 for his work on the photoelectric effect.
He published the general theory of relativity in 1915.
"""

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
docs = [Document(page_content=chunk) for chunk in splitter.split_text(sample_text)]

# ---------------- Embedding + Vector Store ----------------
embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)
vector_db = FAISS.from_documents(docs, embedding_model)

# ---------------- RAG Chain ----------------
retriever = vector_db.as_retriever()
rag_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_key),
    retriever=retriever,
    return_source_documents=True
)

# ---------------- Agent ----------------
# Wrap the RAG chain to return only the result
def rag_tool_func(query: str) -> str:
    return rag_chain({"query": query})["result"]

rag_tool = Tool(
    name="RAGRetriever",
    func=rag_tool_func,
    description="Use this tool to answer questions about Einstein and relativity."
)

agent = initialize_agent(
    tools=[rag_tool],
    llm=ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_key),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ---------------- Run Sample ----------------
if __name__ == "__main__":
    print("\n--- RAG Agent with FAISS ---")
    question = "When did Anup publish general relativity?"
    response = agent.run(question)
    print("\n✅ Answer:", response)
