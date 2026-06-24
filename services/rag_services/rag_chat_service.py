from services.rag_services.retrieval_service import retrieve_chunks
from services.openai_client import client

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using only the context provided below.
If the answer is not in the context, say you don't know. You are an AWS Certified AI Practitioner tutor.
Provide concise exam-focused explanations. Keep responses under 200 words."""


async def rag_chat(question: str) -> str:
    chunks = await retrieve_chunks(question)

    context = "\n\n".join([
        f"[Source: {chunk.document_name}, Page {chunk.page_number}]\n{chunk.text}"
        for chunk in chunks
    ])

    prompt = f"""Context:
{context}

Question: {question}"""

    response = await client.responses.create(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )

    return response.output_text
