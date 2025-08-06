import os
import time
import requests
import fitz  # PyMuPDF
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase

# --- Config ---
AURA_URI = os.getenv("AURA_URI", "neo4j+s://6f9aa9c3.databases.neo4j.io")
AURA_USER = os.getenv("AURA_USER", "neo4j")
AURA_PASSWORD = os.getenv("AURA_PASSWORD", "ZkQ5bYSkAHkajmXH-UnseaZqkKM2HB8c_EKOJEilWHs")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "e0e272cd2f3ac51a8dda3c63908707c12fc63da7d51f0c7a8fbba91e80db3a88")

# --- Init ---
app = FastAPI()
openai.api_key = OPENAI_API_KEY
neo4j_driver = GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PASSWORD))

class EvaluationRequest(BaseModel):
    documents: str
    questions: List[str]

SYSTEM_PROMPT = """
You are an AI assistant for insurance policy documents. For each user query, answer in a single sentence based only on the context provided.

Only include the final answer. Do not repeat the question or say “Based on the context”. No JSON. No lists. Just the sentence.
"""

# --- Helpers ---
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

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def get_embedding(text):
    res = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return res["data"][0]["embedding"]

def get_top_k_chunks(question, chunks, k=3):
    q_emb = get_embedding(question)
    chunk_embs = [get_embedding(chunk) for chunk in chunks]
    sims = cosine_similarity([q_emb], chunk_embs)[0]
    top_k = np.argsort(sims)[-k:][::-1]
    return [chunks[i] for i in top_k]

def log_to_neo4j(question, answer):
    with neo4j_driver.session() as session:
        session.run("""
            CREATE (:QueryLog {
                question: $q,
                answer: $a,
                timestamp: datetime()
            })
        """, q=question, a=answer)

# --- Main Endpoint ---
@app.post("/hackrx/run")
async def handle_request(request: EvaluationRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    if authorization.split(" ")[1] != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    try:
        text = extract_pdf_text(request.documents)
        chunks = chunk_text(text)
        final_answers = []

        for question in request.questions:
            top_chunks = get_top_k_chunks(question, chunks)
            prompt = f"""
[CLAUSES]
{'\n'.join(top_chunks)}

[QUESTION]
{question}
            """

            completion = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )

            answer_text = completion["choices"][0]["message"]["content"].strip()
            final_answers.append(answer_text)
            log_to_neo4j(question, answer_text)

        return {
            "answers": final_answers
        }

    except Exception as e:
        return {
            "answers": [],
            "error": str(e),
            "success": False
        }
