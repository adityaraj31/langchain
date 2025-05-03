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

# schema for the structured output
json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_theme": {
            "type": "string",
            "description": "The name of the theme."
        },
        "summary": {
            "type": "string",
            "description": "A summary of the review."
        },
        "sentiment": {
            "type": "string",
            "description": "The sentiment of the review.",
            "enum": ["positive", "negative", "neutral"]
        },
        "pros": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Pros of the product."
        },
        "cons": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Cons of the product."
        },
        "name": {
            "type": "string",
            "description": "The name of the reviewer."
        },
    },
}

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""The hardware is great, but the software is buggy...""")
print(result)
