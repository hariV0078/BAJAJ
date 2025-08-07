import os
import fitz
import numpy as np
from typing import List
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity
import aiohttp

app = FastAPI()

# ----- Models -----
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# ----- Secure OpenAI Client -----
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----- Get PDF Text from URL -----
async def download_pdf_text(pdf_url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(pdf_url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail="Failed to download PDF.")
            data = await response.read()
            with open("/tmp/temp.pdf", "wb") as f:
                f.write(data)
            doc = fitz.open("/tmp/temp.pdf")
            text = " ".join(page.get_text() for page in doc)
            doc.close()
            return text

# ----- Embedding Function -----
async def get_embedding(text: str) -> List[float]:
    response = await openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# ----- Answer Generator -----
async def generate_answer(context: str, question: str) -> str:
    response = await openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for answering insurance policy-related questions based on document context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]
    )
    return response.choices[0].message.content.strip()

# ----- Vector Similarity Matching -----
async def find_best_context(question: str, chunks: List[str]) -> str:
    question_emb = np.array(await get_embedding(question)).reshape(1, -1)
    chunk_embeddings = [np.array(await get_embedding(chunk)).reshape(1, -1) for chunk in chunks]
    similarities = [cosine_similarity(question_emb, emb)[0][0] for emb in chunk_embeddings]
    best_idx = int(np.argmax(similarities))
    return chunks[best_idx]

# ----- Main Endpoint -----
@app.post("/hackrx/run")
async def run_hackrx(request: HackRxRequest, authorization: str = Header(...)):
    # ✅ Validate Bearer Token
    expected_token = os.getenv("BEARER_TOKEN")  # Set this in Render env
    if authorization != f"Bearer {expected_token}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # ✅ Step 1: Download and parse PDF
    text = await download_pdf_text(request.documents)
    chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]  # Chunking

    # ✅ Step 2: For each question, find best chunk and generate answer
    answers = []
    for question in request.questions:
        context = await find_best_context(question, chunks)
        answer = await generate_answer(context, question)
        answers.append(answer)

    return {"answers": answers}

# Health check
@app.get("/")
async def health():
    return {"status": "running"}
