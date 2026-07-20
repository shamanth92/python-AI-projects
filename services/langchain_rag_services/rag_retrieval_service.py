from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Module-level (created once, reused across requests) since these don't hold
# any state tied to a specific ChromaDB collection snapshot.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-5-mini")

# Two prompt templates share the same retrieval step but produce very
# different output -- this is the "one endpoint, swap the prompt by mode"
# design: qa_prompt answers directly, exam_prompt generates practice questions.
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
Citation: <source from context>

Generate 3 questions. Base the questions only on the context provided, but they should mock actual scenario based questions that you might see in the real exam.
Can you add citations to each question informing the user which part of the context was used to generate the question?
----------------
{context}
----------------
Topic: {question}
""")


# Builds and runs an LCEL chain (LangChain Expression Language -- the `|`
# pipe operator composes runnables into a pipeline):
#   1. {"context": retriever, "question": RunnablePassthrough()}
#      runs the retriever on the input question to fetch relevant chunks,
#      while passing the original question through unchanged, producing
#      {"context": [...chunks], "question": "..."}
#   2. prompt   -> fills the template with that dict
#   3. llm      -> sends the filled prompt to the chat model
#   4. StrOutputParser() -> extracts the plain string answer from the LLM response
#
# vector_store/retriever/chain are (re)built on every call rather than cached
# at module level -- caching them would hold a reference to the ChromaDB
# collection that goes stale after a delete_collection() call (e.g. via the
# DELETE /langchain/ingestion/ endpoint), causing "collection does not exist"
# errors on the next query.
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
