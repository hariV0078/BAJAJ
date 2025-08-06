import os
import fitz  # PyMuPDF
import numpy as np
from typing import List
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from openai import AsyncOpenAI
from neo4j import AsyncGraphDatabase
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
import aiohttp
import logging

# --- Setup logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Environment ---
AURA_URI = os.getenv("AURA_URI")
AURA_USER = os.getenv("AURA_USER")
AURA_PASSWORD = os.getenv("AURA_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# --- Initialize clients ---
app = FastAPI()
ai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

@lru_cache()
def get_neo4j_driver():
    return AsyncGraphDatabase.driver(
        AURA_URI,
        auth=(AURA_USER, AURA_PASSWORD)
    )

# --- Request Schema ---
class EvaluationRequest(BaseModel):
    documents: str
    questions: List[str]

# --- System Prompt ---
SYSTEM_PROMPT = """
You are an AI assistant for insurance policies. Answer each question in one complete sentence using only the context provided.
Do not say "Based on the context" or "According to the document". Do not return JSON or bullet points. Just the answer as a sentence.
"""

# --- PDF Processing ---
async def download_pdf(url: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail="Failed to download PDF")
            return await response.read()

async def extract_pdf_text(pdf_bytes: bytes) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return " ".join(page.get_text() for page in doc)

# --- Text Processing ---
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# --- Embeddings ---
async def get_embedding(text: str) -> List[float]:
    try:
        response = await ai_client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail="Embedding generation failed")

# --- Similarity Search ---
async def get_top_k_chunks(question: str, chunks: List[str], k: int = 3) -> List[str]:
    try:
        q_embedding = await get_embedding(question)
        chunk_embeddings = [await get_embedding(chunk) for chunk in chunks]
        similarities = cosine_similarity([q_embedding], chunk_embeddings)[0]
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [chunks[i] for i in top_indices]
    except Exception as e:
        logger.error(f"Chunk retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail="Chunk retrieval failed")

# --- Answer Generation ---
async def generate_answer(question: str, context: str) -> str:
    try:
        response = await ai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Answer generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Answer generation failed")

# --- Logging ---
async def log_query(question: str, answer: str):
    try:
        driver = get_neo4j_driver()
        async with driver.session() as session:
            await session.run(
                "CREATE (:QueryLog {question: $q, answer: $a, timestamp: datetime()})",
                q=question, a=answer
            )
    except Exception as e:
        logger.error(f"Neo4j logging error: {str(e)}")

# --- Main Endpoint ---
@app.post("/hackrx/run")
async def process_request(
    request: EvaluationRequest,
    authorization: str = Header(None)
):
    # Authentication
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    if authorization.split(" ")[1] != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    try:
        # Download and process PDF
        pdf_bytes = await download_pdf(request.documents)
        full_text = await extract_pdf_text(pdf_bytes)
        chunks = chunk_text(full_text)
        
        # Process questions
        answers = []
        for question in request.questions:
            try:
                relevant_chunks = await get_top_k_chunks(question, chunks)
                context = "\n".join(relevant_chunks)
                answer = await generate_answer(question, context)
                answers.append(answer)
                await log_query(question, answer)
            except Exception as e:
                answers.append(f"Error processing question: {str(e)}")
                continue

        return {"answers": answers, "success": True}

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
