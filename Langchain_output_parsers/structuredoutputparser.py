from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    model_name="llama3-8b-8192",
)

schema = [
    ResponseSchema(name='fact1', description='Fact 1 about the topic.'),
    ResponseSchema(name='fact2', description='Fact 2 about the topic.'),
    ResponseSchema(name='fact3', description='Fact 3 about the topic.'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 facts about {topic}\n{format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = template | llm | parser

final_result = chain.invoke({"topic": "Python programming language"})

print(final_result)  # Parsed result as a dictionary
