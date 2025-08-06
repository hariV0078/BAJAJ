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

# --- Environment ---
AURA_URI = os.getenv("AURA_URI")
AURA_USER = os.getenv("AURA_USER")
AURA_PASSWORD = os.getenv("AURA_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# --- Init ---
app = FastAPI()
client = OpenAI(api_key=OPENAI_API_KEY)

@lru_cache()
def get_neo4j_driver():
    return GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PASSWORD))

# --- Request Schema ---
class EvaluationRequest(BaseModel):
    documents: str
    questions: List[str]

# --- System Prompt ---
SYSTEM_PROMPT = """
You are an AI assistant for insurance policies. Answer each question in one complete sentence using only the context provided.

Do not say “Based on the context” or “According to the document”. Do not return JSON or bullet points. Just the answer as a sentence.
"""

# --- PDF Text Extraction ---
def extract_pdf_text(url: str) -> str:
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download PDF.")
    with open("temp_policy.pdf", "wb") as f:
        f.write(response.content)
    text = ""
    with fitz.open("temp_policy.pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# --- Chunking ---
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# --- Embedding ---
def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# --- Top-k Chunk Retrieval ---
def get_top_k_chunks(question: str, chunks: List[str], k=3) -> List[str]:
    q_emb = get_embedding(question)
    chunk_embs = [get_embedding(chunk) for chunk in chunks]
    sims = cosine_similarity([q_emb], chunk_embs)[0]
    top_k_indices = np.argsort(sims)[-k:][::-1]
    return [chunks[i] for i in top_k_indices]

# --- Logging ---
def log_to_neo4j(question: str, answer: str):
    with get_neo4j_driver().session() as session:
        session.run(
            "CREATE (:QueryLog {question: $q, answer: $a, timestamp: datetime()})",
            q=question, a=answer
        )

# --- GPT Completion ---
def get_gpt_answer(prompt: str) -> str:
    chat = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return chat.choices[0].message.content.strip()

# --- Main Endpoint ---
@app.post("/hackrx/run")
async def handle_request(request: EvaluationRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    if authorization.split(" ")[1] != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    try:
        full_text = extract_pdf_text(request.documents)
        chunks = chunk_text(full_text)
        final_answers = []

        for question in request.questions:
            top_chunks = get_top_k_chunks(question, chunks, k=3)
            prompt = f"[CLAUSES]\n{chr(10).join(top_chunks)}\n\n[QUESTION]\n{question}"
            answer = get_gpt_answer(prompt)
            log_to_neo4j(question, answer)
            final_answers.append(answer)

        return {
            "answers": final_answers,
            "success": True
        }

    except Exception as e:
        return {
            "answers": [],
            "error": str(e),
            "success": False
        }
