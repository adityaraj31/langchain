from langchain_openai import OpeonAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()""

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", diemension=300)

documents = [
    "Virat kholi is a famous Indian cricketer known for his aggressive batting style and leadership skills.",
    "MS Dhoni is a legendary Indian cricketer known for his calm demeanor and exceptional wicketkeeping skills.",
    "Sachin Tendulkar is regarded as one of the greatest batsmen in the history of cricket, with numerous records to his name.",
    "Rohit Sharma is known for his elegant batting style and has captained the Indian cricket team in various formats.",
    "Jasprit Bumrah is a fast bowler known for his unique bowling action and ability to bowl yorkers consistently.",
]

query = "tell me about virat kholi"

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print(f"Score: {score}")