from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-5-mini")

qa_prompt = ChatPromptTemplate.from_template("""
You are an AWS Certified AI Practitioner tutor. Answer the question using only the context below.
If the answer is not in the context, say you don't know.
Provide concise exam-focused explanations. Keep responses under 500 words.
----------------
{context}
----------------
Question: {question}
""")

exam_prompt = ChatPromptTemplate.from_template("""
You are an AWS Certified AI Practitioner exam coach. Using the context below, generate exam-style \
multiple choice questions on the topic the user asks about.

Format each question as:
Q: <question>
A) ...
B) ...
C) ...
D) ...
Answer: <correct option>
Explanation: <brief explanation>

Generate 3 questions. Base them only on the context provided.
----------------
{context}
----------------
Topic: {question}
""")


async def generate_answer(question: str, mode: str = "qa") -> str:
    prompt = exam_prompt if mode == "exam" else qa_prompt  # mode value is the enum's string value
    vector_store = Chroma(
        collection_name="aws-rag-documents",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return await chain.ainvoke(question)
