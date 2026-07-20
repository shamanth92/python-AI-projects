from services.rag_services.retrieval_service import retrieve_chunks
from services.openai_client import client

# System prompt tells the model to act as an AWS exam tutor and to only use
# the retrieved context -- this is what keeps answers grounded in the ingested
# docs instead of the model's general knowledge (reduces hallucination).
SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using only the context provided below.
If the answer is not in the context, say you don't know. You are an AWS Certified AI Practitioner tutor.
Provide concise exam-focused explanations. Keep responses under 200 words."""


# Final step of the "basic" RAG pipeline: retrieve relevant chunks for the
# question, stitch them into a single context block (tagged with source file
# + page so the model could cite them if asked), and send question + context
# to the chat model.
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
