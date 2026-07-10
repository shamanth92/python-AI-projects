from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the user's question using only the context provided below.
If the answer is not in the context, say you don't know. You are an AWS Certified AI Practitioner tutor.
Provide concise exam-focused explanations. Keep responses under 500 words.
----------------
{context}
----------------
Question: {question}
""")


async def generate_answer(question: str) -> str:
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
