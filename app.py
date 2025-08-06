import os
import requests
import fitz  # PyMuPDF
from openai import OpenAI
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
openai_client = OpenAI(api_key=OPENAI_API_KEY)
neo4j_driver = GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PASSWORD))

class EvaluationRequest(BaseModel):
    documents: str
    questions: List[str]

SYSTEM_PROMPT = """
You are an AI assistant for insurance policies. Answer each question in one complete sentence using only the context provided.

Do not say “Based on the context” or “According to the document”. Do not return JSON or bullet points. Just the answer as a sentence.
"""

# --- Step 1: Extract PDF text from the blob URL ---
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

# --- Step 2: Store clauses in Neo4j (if needed) ---
def store_clauses_in_neo4j(clauses: List[str]):
    with neo4j_driver.session() as session:
        for i, clause in enumerate(clauses):
            session.run(
                "MERGE (c:Clause {id: $id}) SET c.text = $text",
                id=f"Clause_{i}", text=clause
            )

# --- Step 3: Get most relevant clauses from Neo4j ---
def retrieve_relevant_clauses(question: str, limit=3):
    with neo4j_driver.session() as session:
        results = session.run(
            """
            CALL db.index.fulltext.queryNodes('clauseIndex', $question) YIELD node, score
            RETURN node.text AS clause
            ORDER BY score DESC
            LIMIT $limit
            """, question=question, limit=limit
        )
        return [record["clause"] for record in results]

# --- Step 4: Log query and answer to Neo4j ---
def log_to_neo4j(question: str, answer: str):
    with neo4j_driver.session() as session:
        session.run(
            "CREATE (:QueryLog {question: $q, answer: $a, timestamp: datetime()})",
            q=question, a=answer
        )

# --- Main Endpoint ---
@app.post("/hackrx/run")
async def handle_request(request: EvaluationRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    if authorization.split(" ")[1] != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    try:
        # Extract and store clauses (if not already stored)
        full_text = extract_pdf_text(request.documents)
        with neo4j_driver.session() as session:
            clause_ids = [r["id"] for r in session.run("MATCH (c:Clause) RETURN c.id AS id")]
        if "Clause_0" not in clause_ids:
            clause_list = full_text.split("\n\n")
            store_clauses_in_neo4j(clause_list)

        final_answers = []
        for question in request.questions:
            top_clauses = retrieve_relevant_clauses(question)
            prompt = f"""
[CLAUSES]
{chr(10).join(top_clauses)}

[QUESTION]
{question}
            """

            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )

            answer = completion.choices[0].message.content.strip()
            final_answers.append(answer)
            log_to_neo4j(question, answer)

        return {
            "answers": final_answers
        }

    except Exception as e:
        return {
            "answers": [],
            "error": str(e),
            "success": False
        }
