import os
import requests
import fitz  # PyMuPDF
import json
from typing import List
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase
from functools import lru_cache

# --- Config ---
AURA_URI = os.getenv("AURA_URI")
AURA_USER = os.getenv("AURA_USER")
AURA_PASSWORD = os.getenv("AURA_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# --- Init ---
app = FastAPI()

@lru_cache()
def get_neo4j_driver():
    return GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PASSWORD))

class EvaluationRequest(BaseModel):
    documents: str
    questions: List[str]

SYSTEM_PROMPT = """
You are an AI assistant for insurance policies. Answer each question in one complete sentence using only the context provided.

Do not say “Based on the context” or “According to the document”. Do not return JSON or bullet points. Just the answer as a sentence.
"""

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

@app.get("/")
def health():
    return {"status": "ok"}

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
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += " " + para
        else:
            chunks.append(current.strip())
            current = para
    if current:
        chunks.append(current.strip())
    return chunks

# --- Store Chunks in Neo4j ---
def store_chunks_in_neo4j(chunks: List[str]):
    with get_neo4j_driver().session() as session:
        for i, chunk in enumerate(chunks):
            session.run(
                "MERGE (c:Clause {id: $id}) SET c.text = $text",
                id=f"Clause_{i}", text=chunk
            )

# --- Query relevant clauses ---
def retrieve_relevant_clauses(question: str, limit=3):
    with get_neo4j_driver().session() as session:
        results = session.run(
            """
            CALL db.index.fulltext.queryNodes('clauseIndex', $question) YIELD node, score
            WHERE size(node.text) < 1000
            RETURN substring(node.text, 0, 500) AS clause
            ORDER BY score DESC
            LIMIT $limit
            """, question=question, limit=limit
        )
        return [record["clause"] for record in results]

# --- Log interaction ---
def log_to_neo4j(question: str, answer: str):
    with get_neo4j_driver().session() as session:
        session.run(
            "CREATE (:QueryLog {question: $q, answer: $a, timestamp: datetime()})",
            q=question, a=answer
        )

# --- OpenAI Completion ---
def get_gpt_answer(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    response = requests.post(OPENAI_API_URL, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(f"OpenAI error: {response.text}")
    return response.json()["choices"][0]["message"]["content"].strip()

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

        # Store only once
        with get_neo4j_driver().session() as session:
            existing = session.run("MATCH (c:Clause) RETURN c.id AS id")
            if not any(r["id"] == "Clause_0" for r in existing):
                store_chunks_in_neo4j(chunks)

        final_answers = []
        for question in request.questions:
            clauses = retrieve_relevant_clauses(question)
            prompt = f"[CLAUSES]\n{chr(10).join(clauses)}\n\n[QUESTION]\n{question}"
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
