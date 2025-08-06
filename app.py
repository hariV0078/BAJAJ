import os
import time
import requests
import fitz  # PyMuPDF
from typing import List
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import openai
from neo4j import GraphDatabase

# --- Configuration ---
AURA_URI = os.getenv("AURA_URI", "neo4j+s://6f9aa9c3.databases.neo4j.io")
AURA_USER = os.getenv("AURA_USER", "neo4j")
AURA_PASSWORD = os.getenv("AURA_PASSWORD", "ZkQ5bYSkAHkajmXH-UnseaZqkKM2HB8c_EKOJEilWHs")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "e0e272cd2f3ac51a8dda3c63908707c12fc63da7d51f0c7a8fbba91e80db3a88")

# --- Initialize FastAPI & Clients ---
app = FastAPI()
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
neo4j_driver = GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PASSWORD))

# --- Request Body Model ---
class EvaluationRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

# --- System Prompt ---
SYSTEM_PROMPT = """
You are an AI policy assistant. Your task is to extract concise, clear answers from an insurance policy document for the user’s query.

Only answer based on the provided content. Respond in a single sentence, and avoid generic disclaimers. Be accurate and specific.

Respond with just the final answer as a plain sentence.
"""

# --- Helper: Extract PDF text from URL ---
def extract_text_from_pdf_url(url: str) -> str:
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

# --- Optional: Log the query into Neo4j (if needed) ---
def log_query_to_neo4j(question: str, answer: str):
    with neo4j_driver.session() as session:
        session.run(
            "CREATE (q:Question {text: $question, answer: $answer, timestamp: datetime()})",
            question=question,
            answer=answer
        )

# --- Main Evaluation Endpoint ---
@app.post("/hackrx/run")
async def handle_request(request: EvaluationRequest, authorization: str = Header(None)):
    start_time = time.time()

    # Auth check
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    if authorization.split(" ")[1] != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    try:
        # 1. Extract context from PDF
        context = extract_text_from_pdf_url(request.documents)

        # 2. Process questions via LLM
        answers = []
        for question in request.questions:
            prompt = f"""
            [DOCUMENT]
            {context}

            [QUESTION]
            {question}
            """

            completion = openai_client.chat.completions.create(
                model="gpt-4o",  # or "o3" if required
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )

            answer = completion.choices[0].message.content.strip()
            answers.append(answer)

            # Optional: log to Neo4j
            log_query_to_neo4j(question, answer)

        return {
            "answers": answers,
            "processing_time": f"{time.time() - start_time:.2f}s"
        }

    except Exception as e:
        return {
            "answers": [],
            "error": str(e),
            "success": False
        }
