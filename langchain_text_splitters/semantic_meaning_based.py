from langchain_experimental.text_splitter import SemanticChunker
from langchain_groq.embeddings import GroqEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    GroqEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=0.5,
)

sample = """
The solar system consists of the Sun and all the objects that orbit around it. 
The Sun is the central star that provides light and heat to the planets.
Mercury is the closest planet to the Sun and has extreme temperature variations.
Venus is often called Earth's sister planet due to their similar sizes.
Earth is the only known planet to support life and has one natural satellite, the Moon.
Mars is known as the Red Planet and has been the focus of many exploration missions.
Jupiter is the largest planet and has a distinctive Great Red Spot storm system.
Saturn is famous for its beautiful rings made mostly of ice and rock particles.
Uranus and Neptune are ice giants located in the outer solar system.
Pluto, once considered the ninth planet, is now classified as a dwarf planet.
"""

docs = text_splitter.create_documents([sample])

print(len(docs))
print(docs)