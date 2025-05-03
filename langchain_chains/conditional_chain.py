from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model_name="llama3-8b-8192",
)

# Define feedback output model
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="The sentiment of the feedback.")

# Output parsers
parser = StrOutputParser()  # ← ADD THIS
parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Prompt for classifying sentiment
prompt1 = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback.\n\n"
        "ONLY output the result in the following format.\n"
        "Do not add any explanation or extra words.\n\n"
        "{format_instructions}\n\n"
        "Feedback:\n{feedback}"
    ),
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()},
)

classifier_chain = prompt1 | llm | parser2

# Prompts for generating a response
prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback: \n{feedback}",
    input_variables=['feedback'],
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback: \n{feedback}",
    input_variables=['feedback'],
)

# Branching based on sentiment
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | llm | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | llm | parser),
    RunnableLambda(lambda x: "No feedback provided."),
)

# Final chain
chain = classifier_chain | branch_chain

# Test
print(chain.invoke({'feedback': "This is a hate product!"}))

chain.get_graph().print_ascii()