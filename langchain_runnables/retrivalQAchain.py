from langchain_community.document_loaders import TextLoader  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# 1. Load the document
loader = TextLoader('sample.txt')
documents = loader.load()

# 2. Split the text into small chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 3. Convert text into embeddings & store in FAISS (Vector Store)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)

# 4. Create a retriever
retriever = vectorstore.as_retriever()

# 5. Initialize the llm
llm = ChatGroq(model_name="llama3-8b-8192")

# 6. Create RetrievalQA chain
from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# 7. Ask the question
query = "What is this document about?"
response = chain.invoke(query)

print(response)