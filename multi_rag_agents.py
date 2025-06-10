import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain_openai import ChatOpenAI

# Load OpenAI API key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_key)
embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)

# Helper function to load → split → embed → store → build retriever
def load_split_embed(loader, label):
    print(f"📄 Loading: {label}")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"🔢 {label} chunks: {len(chunks)}")
    db = FAISS.from_documents(chunks, embedding_model)
    retriever = db.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    return Tool(name=f"{label}Agent", func=chain.run, description=f"Answer questions using {label} content.")

# 1. TXT Agent
txt_loader = TextLoader("sample.txt")
txt_tool = load_split_embed(txt_loader, "TXT")

# 2. PDF Agent
pdf_loader = PyPDFLoader("sample.pdf")
pdf_tool = load_split_embed(pdf_loader, "PDF")

# 3. DOCX Agent
docx_loader = Docx2txtLoader("sample.docx")
docx_tool = load_split_embed(docx_loader, "DOCX")

# 4. Webpage Agent
web_loader = WebBaseLoader("https://en.wikipedia.org/wiki/Albert_Einstein")
web_tool = load_split_embed(web_loader, "WEB")

# Export tools as dictionary
tools = {
    "txt": txt_tool,
    "pdf": pdf_tool,
    "docx": docx_tool,
    "web": web_tool
}