from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a five pointer summary from the following text \n {text}",
    input_variables=["text"]
)

llm = ChatGroq(
    model_name="llama3-8b-8192",
)

parser = StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm | parser

result = chain.invoke({'topic': "Unemployment in India"})

print(result)

chain.get_graph().print_ascii()