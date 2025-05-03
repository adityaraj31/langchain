from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()   

llm1 = ChatGroq(
    model_name="llama3-8b-8192",
)

llm2 = ChatGroq(
    model_name="llama3-70b-8192",
)

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n {notes} \n {quiz}",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | llm1 | parser,
    'quiz': prompt2 | llm2 | parser,
})

merge_chain = prompt3 | llm1 | parser

chain = parallel_chain | merge_chain

text = """
Artificial intelligence (AI) is a branch of computer science that aims to create systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, understanding natural language, and perceiving the environment. AI has been a transformative force in various industries, including healthcare, finance, transportation, and entertainment.

In healthcare, AI is used for diagnosing diseases, personalizing treatment plans, and even predicting patient outcomes. For example, machine learning algorithms can analyze medical images to detect conditions like cancer at an early stage. In finance, AI powers fraud detection systems, algorithmic trading, and personalized financial advice. Autonomous vehicles, which rely heavily on AI, are revolutionizing the transportation industry by improving safety and efficiency.

Despite its many benefits, AI also raises ethical and societal concerns. Issues such as data privacy, algorithmic bias, and the potential for job displacement are topics of ongoing debate. As AI continues to evolve, it is crucial to address these challenges to ensure that the technology is used responsibly and equitably.

The future of AI holds immense potential, with advancements in areas like natural language processing, robotics, and quantum computing. As researchers and developers push the boundaries of what AI can achieve, it is essential to balance innovation with ethical considerations to create a positive impact on society.
"""

result = chain.invoke({'text': text})

print(result)

chain.get_graph().print_ascii()