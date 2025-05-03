from langchain_core.prompts import PromptTemplate

#template
template = PromptTemplate(
    template="""
    Please summarize the research paper titled "{paper_input}" with the following specifications:
    Explanation Style: {style_input}
    Explanation Length: {length_input}
    1. Mathematical Details:
        - Include relevant mathematical equations if present in the paper.
        - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
    2. Analogies:
        - Use analogies to explain complex concepts in a relatable manner.
        - Provide examples that are easy to understand for someone with a basic understanding of the topic.
    If certain information is not available in the paper, respond with "Not Available".
    Ensure the summary is clear, concise, and easy to understand for someone with a basic understanding of the topic.
    """,
    input_variables=["paper_input", "style_input", "length_input"],
)

template.save('template.json')