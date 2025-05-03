from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatOpenAI(
    openai_api_key=groq_api_key,
    openai_api_base="https://api.groq.com/openai/v1",
    model_name="llama3-70b-8192"
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about langchain."),
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages) 