from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

text = """# Machine Learning Basics

## Introduction to Neural Networks
Neural networks are computational models inspired by biological neural networks. They consist of layers of interconnected nodes or "neurons" that process information.

### Types of Neural Networks
- Feedforward Neural Networks
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)

## Training Process
The training process involves:
1. Forward propagation
2. Loss calculation 
3. Backpropagation

### Optimization Algorithms
Common optimization algorithms include:
- Gradient Descent
- Adam
- RMSprop

## Applications
Neural networks have various applications in:
- Computer Vision
- Natural Language Processing
- Speech Recognition
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=100,
    chunk_overlap=0,
)

chunks = splitter.create_documents([text])

print(len(chunks))
print(chunks)