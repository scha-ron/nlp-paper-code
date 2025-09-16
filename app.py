from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Path to Corpus PDF
# PDF_PATH = ""

# Path to your chroma database
# CHROMA_PATH = ""

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
llm = ChatOllama(model="qwen3:1.7b")  # Ensure qwen3 is available in your Ollama

# Build RAG chain
def build_rag():
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

# === Interactive CLI ===
if __name__ == "__main__":
    qa = build_rag()
    print("Ask me anything about studying CS in Germany as a Ugandan student. Type 'exit' to quit.\n")

    while True:
        user_input = input("Your question: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        response = qa.invoke(user_input)
        answer = response.get("result", "") or response.get("text", "") or ""
        print(f"Answer:\n{answer.strip()}\n{'-'*60}")


