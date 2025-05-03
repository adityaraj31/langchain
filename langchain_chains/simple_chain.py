from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"],
)

llm = ChatGroq(
    model_name="llama3-8b-8192",
)

parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({"topic": "Python programming"})

print(result)

chain.get_graph().print_ascii()