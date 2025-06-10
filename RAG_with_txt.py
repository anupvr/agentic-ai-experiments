import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.agents import Tool

from langchain_openai import ChatOpenAI

# Load API key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# ---------------- Step 1: Load TXT File ----------------
loader = TextLoader("sample.txt")  # Replace with your actual file
documents = loader.load()

# ---------------- Step 2: Split into Chunks ----------------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# ---------------- Step 3: Embedding + Vector DB ----------------
embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)
vector_db = FAISS.from_documents(chunks, embedding_model)
retriever = vector_db.as_retriever()

# ---------------- Step 4: RAG Chain ----------------
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_key)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ---------------- Step 5: Tool for LangChain Agent ----------------

def txt_tool_func(query: str) -> str:
    return rag_chain({"query": query})["result"]

txt_tool = Tool(
    name="TXTAgent",
    func=txt_tool_func,
    description="Use this to answer questions based on the sample.txt document."
)
# Optional standalone run
if __name__ == "__main__":
    query = "What did the document say about Einstein?"
    print("\n🧪 Running TXT RAG Tool:")
    print(txt_tool.run(query))