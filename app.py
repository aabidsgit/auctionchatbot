import os
import sys
import streamlit as st
from dotenv import load_dotenv
from tqdm import tqdm

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
# Fix SQLite version issue for Chroma
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

# Load environment variables
load_dotenv()
os.environ["CHROMA_TELEMETRY"] = "False"

# Langchain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Streamlit UI
st.title("Auction.com Chatbot")
st.write("This chatbot is trained on foreclosure documents provided by Mike's Team.")

# File and DB paths
PDF_PATH = "Foreclosure_Full_Doc.pdf"
CHROMA_PATH = "./chroma_db"

# Main function to load or build retriever
@st.cache_resource(show_spinner="Loading or building vector DB...")
def load_or_build_retriever():
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        loader = PyPDFLoader(PDF_PATH)
        data = loader.load()

        # Chunking: smaller to reduce API timeout
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(data)

        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]

        vectorstore = Chroma(
            collection_name="auction_docs",
            embedding_function=embedding_model,
            persist_directory=CHROMA_PATH
        )

        # Embed in batches to avoid timeout
        batch_size = 50
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            vectorstore.add_texts(texts=batch_texts, metadatas=batch_metas)

        #vectorstore.persist()
        st.success("✅ Vectorstore built and saved.")
    else:
        vectorstore = Chroma(
            collection_name="auction_docs",
            embedding_function=embedding_model,
            persist_directory=CHROMA_PATH
        )

    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Load retriever
retriever = load_or_build_retriever()

# LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1)

# Prompt setup
system_prompt = (
    "You are an assistant for answering questions posed by auction.com employees. Ensure that all answers are related to Auction.com data "
    "provided to you. If you don't know the answer, say that you don't know and need more data. Be detailed and thorough.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)

# Main chat input + output
query = st.chat_input("Hello Auction.com Team! Go ahead and ask something")
if query:
    with st.spinner("Thinking..."):
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": query})

        st.write(f"*User Asked:* {query}")
        st.write(f"*Chatbot Response:* {response['answer']}")
