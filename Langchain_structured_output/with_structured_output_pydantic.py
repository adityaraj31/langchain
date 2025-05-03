from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")  # Make sure this is set!

model = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo"  # ✅ Switched to OpenAI model
)

class Review(BaseModel):
    key_themes: list[str] = Field(description="Key themes in the review")
    summary: str = Field(description="Summary of the review")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Sentiment of the review")
    pros: Optional[list[str]] = Field(description="Pros of the product", default= None)
    cons: Optional[list[str]] = Field(description="Cons of the product", default= None)
    name: Optional[str] = Field(description="Name of the reviewer", default= None)

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software is buggy...""")
print(result)
