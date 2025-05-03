from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Hugging Face endpoint
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
)

# Wrap the LLM in a chat interface
model = ChatHuggingFace(llm=llm)

# Invoke the model with a structured input
result = model.invoke([{"role": "user", "content": "what is the capital of India?"}])

# Print the result
print(result.content)  # Adjust based on the actual structure of `result`