# rag_system.py

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Path to your PDF
PDF_PATH = r"C:\Users\rizaa\OneDrive\Desktop\IBS\SEM6\NLP\nlp_bot_2\data\qa_corpus.pdf"
CHROMA_PATH = "chroma_store"

# Load documents
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)

# Create embeddings
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Setup Chroma vector store
db = Chroma.from_documents(docs, embedding=embedder, persist_directory=CHROMA_PATH)
retriever = db.as_retriever()

# Define LLM
llm = OllamaLLM(model="qwen3:1.7b")  # Make sure this is installed in Ollama

# Build QA chain
def build_rag(ground_truths=None):
    prompt_template = PromptTemplate.from_template(
        """
        You are a helpful AI assistant that offers relevant and up-to-date guidance for Ugandans seeking to study computer science
        in Germany. Answer the following question based only on the provided context. If you don't know the answer, say so. Be concise and factual.

Context:
{context}

Question:
{question}

Answer:"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=False,
    )

    return qa_chain
