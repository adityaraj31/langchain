from langchain_community.document_loaders import TextLoader  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Step 0: Load environment variables
load_dotenv()

# 1. Load the document
loader = TextLoader('sample.txt')  # You need a 'sample.txt' file
documents = loader.load()

# 2. Split the text into small chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 3. Convert text into embeddings & store in FAISS
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)

# 4. Create a retriever
retriever = vectorstore.as_retriever()

# 5. Manually Retrieve Relevant document
query = "What is this document about?"
retrieved_docs = retriever.invoke(query)  # Updated method

# 6. Combine Retrieved text into a single prompt
combined_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 7. Initialize the llm
llm = ChatGroq(model_name="llama3-8b-8192")

# 8. Manually pass the Retrieved text into llm
final_prompt = f"""Use the following information to answer the question:

{combined_text}

Question: {query}
Answer:"""

response = llm.invoke(final_prompt)

# 9. Print the answer
print("\nAnswer:", response.content)
