from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=1.7,
)

class Person(BaseModel):

    name: str = Field(description="The name of the person")
    age: int = Field(gt=18, description="The age of the person")
    city: str = Field(description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = "Generate the name, age and city of a fictional {place} person \n {format_instructions}",
    input_variables = ["place"],
    partial_variables = {"format_instructions": parser.get_format_instructions()}
)

# prompt = template.format(place="New York")

# print(prompt)

# result = llm.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)

chain = template | llm | parser

final_result = chain.invoke({"place": "India"})

print(final_result)