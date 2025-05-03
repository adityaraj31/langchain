from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    model_name="llama3-8b-8192",  # or "mixtral-8x7b-32768", etc.
)

parser = JsonOutputParser();

template = PromptTemplate(
    template="Give me 5 facts about {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={
        "format_instruction": parser.get_format_instructions(),
    }
)

# prompt = template.format()

# result = llm.invoke(prompt)

# final_result = parser.parse(result.content)

chain = template | llm | parser 

resutl = chain.invoke({'topic': 'Black holes'})

print(resutl)