import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Fetch the API key from environment variable
google_api_key = os.getenv("GOOGLE_API_KEY")

# Ensure that the API key is loaded
if google_api_key is None:
    raise ValueError("API key not found. Please check your .env file.")

# Initialize the model
model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro-latest",  # ✅ Use a supported model name
    google_api_key=google_api_key,        # ✅ Use API key from environment variable
)

# Invoke the model and print the result
result = model.invoke("What is the capital of India?")
print(result.content)
