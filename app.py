import os
import requests
import fitz  # PyMuPDF
import json
import numpy as np
from typing import List
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from neo4j import GraphDatabase
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
import aiohttp
import asyncio
from fastapi.middleware.cors import CORSMiddleware

# --- Environment ---
AURA_URI = os.getenv("AURA_URI")
AURA_USER = os.getenv("AURA_USER")
AURA_PASSWORD = os.getenv("AURA_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# Timeout settings (in seconds)
PDF_DOWNLOAD_TIMEOUT = 30
OPENAI_TIMEOUT = 30
NEO4J_TIMEOUT = 10

# --- Init ---
app = FastAPI()

# Add CORS middleware to handle potential CORS issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure HTTP client with timeouts
client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=OPENAI_TIMEOUT
)

@lru_cache()
def get_neo4j_driver():
    return GraphDatabase.driver(
        AURA_URI, 
        auth=(AURA_USER, AURA_PASSWORD),
        connection_timeout=NEO4J_TIMEOUT
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

# --- PDF Text Extraction ---
async def extract_pdf_text(url: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=PDF_DOWNLOAD_TIMEOUT) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download PDF. Status: {response.status}")
                
                content = await response.read()
                with open("temp_policy.pdf", "wb") as f:
                    f.write(content)
                
                text = ""
                with fitz.open("temp_policy.pdf") as doc:
                    for page in doc:
                        text += page.get_text()
                return text
    except Exception as e:
        raise Exception(f"PDF processing error: {str(e)}")

# --- Chunking ---
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# --- Embedding ---
async def get_embedding(text: str) -> List[float]:
    try:
        response = await asyncio.wait_for(
            client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            ),
            timeout=OPENAI_TIMEOUT
        )
        return response.data[0].embedding
    except asyncio.TimeoutError:
        raise Exception("OpenAI embedding request timed out")

# --- Top-k Chunk Retrieval ---
async def get_top_k_chunks(question: str, chunks: List[str], k=3) -> List[str]:
    try:
        q_emb = await get_embedding(question)
        chunk_embs = [await get_embedding(chunk) for chunk in chunks]
        sims = cosine_similarity([q_emb], chunk_embs)[0]
        top_k_indices = np.argsort(sims)[-k:][::-1]
        return [chunks[i] for i in top_k_indices]
    except Exception as e:
        raise Exception(f"Chunk retrieval error: {str(e)}")

# --- Logging ---
async def log_to_neo4j(question: str, answer: str):
    try:
        driver = get_neo4j_driver()
        async with driver.session() as session:
            await session.run(
                "CREATE (:QueryLog {question: $q, answer: $a, timestamp: datetime()})",
                q=question, a=answer
            )
    except Exception as e:
        print(f"Neo4j logging failed (non-critical): {str(e)}")

# --- GPT Completion ---
async def get_gpt_answer(prompt: str) -> str:
    try:
        chat = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            ),
            timeout=OPENAI_TIMEOUT
        )
        return chat.choices[0].message.content.strip()
    except asyncio.TimeoutError:
        raise Exception("OpenAI chat completion timed out")

# --- Main Endpoint ---
@app.post("/hackrx/run")
async def handle_request(request: EvaluationRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    if authorization.split(" ")[1] != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    try:
        full_text = await extract_pdf_text(request.documents)
        chunks = chunk_text(full_text)
        final_answers = []

        for question in request.questions:
            try:
                top_chunks = await get_top_k_chunks(question, chunks, k=3)
                prompt = f"[CLAUSES]\n{chr(10).join(top_chunks)}\n\n[QUESTION]\n{question}"
                answer = await get_gpt_answer(prompt)
                asyncio.create_task(log_to_neo4j(question, answer))  # Fire and forget
                final_answers.append(answer)
            except Exception as e:
                final_answers.append(f"Error processing question: {str(e)}")

        return {
            "answers": final_answers,
            "success": True
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "success": False
            }
        )
