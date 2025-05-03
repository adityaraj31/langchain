from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableSequence

load_dotenv()

def word_count(text):
    return len(text.split(" "))

prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"],
)

model = ChatGroq(
    model="llama3-8b-8192"
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "word_count": RunnableLambda(word_count)
    }
)

final_chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain
)

print(final_chain.invoke("bears"))