from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")  # Make sure this is set!

model = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo"  # ✅ Switched to OpenAI model
)

class Review(TypedDict):
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software is buggy...""")
print(result)
