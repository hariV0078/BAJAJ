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

# --- Request Model ---
class EvaluationRequest(BaseModel):
    documents: str
    questions: List[str]

# --- Constants ---
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
SYSTEM_PROMPT = """
You are an AI assistant for insurance policies. Answer each question in one complete sentence using only the context provided.

Do not say “Based on the context” or “According to the document”. Do not return JSON or bullet points. Just the answer as a sentence.
"""

# --- Health Check ---
@app.get("/")
def health():
    return {"status": "ok"}

# --- PDF Extraction ---
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

# --- Neo4j Functions ---
def store_clauses_in_neo4j(clauses: List[str]):
    with get_neo4j_driver().session() as session:
        for i, clause in enumerate(clauses):
            session.run(
                "MERGE (c:Clause {id: $id}) SET c.text = $text",
                id=f"Clause_{i}", text=clause
            )

def retrieve_relevant_clauses(question: str, limit=3):
    with get_neo4j_driver().session() as session:
        results = session.run(
            """
            CALL db.index.fulltext.queryNodes('clauseIndex', $question) YIELD node, score
            RETURN node.text AS clause
            ORDER BY score DESC
            LIMIT $limit
            """, question=question, limit=limit
        )
        return [record["clause"] for record in results]

def log_to_neo4j(question: str, answer: str):
    with get_neo4j_driver().session() as session:
        session.run(
            "CREATE (:QueryLog {question: $q, answer: $a, timestamp: datetime()})",
            q=question, a=answer
        )

# --- OpenAI Call via HTTP ---
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
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()

# --- Main Endpoint ---
@app.post("/hackrx/run")
async def handle_request(request: EvaluationRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    if authorization.split(" ")[1] != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    try:
        full_text = extract_pdf_text(request.documents)

        # Store clauses only if not already present
        existing = [r["id"] for r in get_neo4j_driver().session().run("MATCH (c:Clause) RETURN c.id AS id")]
        if "Clause_0" not in existing:
            clause_list = full_text.split("\n\n")
            store_clauses_in_neo4j(clause_list)

        final_answers = []
        for question in request.questions:
            top_clauses = retrieve_relevant_clauses(question)
            prompt = f"[CLAUSES]\n{chr(10).join(top_clauses)}\n\n[QUESTION]\n{question}"
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
