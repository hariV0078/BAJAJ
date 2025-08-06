import os
import time
from fastapi import FastAPI, Request, Header, HTTPException
from neo4j import GraphDatabase
import openai
from pydantic import BaseModel

# --- Configuration ---
AURA_URI = os.getenv("AURA_URI", "neo4j+s://6f9aa9c3.databases.neo4j.io")
AURA_USER = os.getenv("AURA_USER", "neo4j")
AURA_PASSWORD = os.getenv("AURA_PASSWORD", "ZkQ5bYSkAHkajmXH-UnseaZqkKM2HB8c_EKOJEilWHs")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "mysecrettoken")  # <-- Add your token here

# --- Initialize Clients ---
app = FastAPI()
neo4j_driver = GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PASSWORD))
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Request Model ---
class QueryRequest(BaseModel):
    user_query: str

# --- System Prompt ---
SYSTEM_PROMPT = """
You are an AI policy assistant. Your only task is to answer a user's query directly and concisely based strictly on the provided context clauses.

Begin your answer with a clear "Yes" or "No". Then, in the same sentence, briefly state the reason. Do not explain your reasoning process or mention the clause numbers.

Please respond in JSON format with the following structure:
{
  "answer": "Yes/No, brief reason here"
}
"""

# --- Main Endpoint ---
@app.post("/hackrx/run")
async def process_query(request: QueryRequest, authorization: str = Header(None)):
    start_time = time.time()

    # 1. Verify Bearer Token
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.split(" ")[1]
    if token != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    try:
        # 2. Retrieve context from Neo4j (this is a placeholder)
        context_clauses = "Clause 3.2: A waiting period of 24 months is applicable for joint replacement surgery..."

        # 3. Construct the prompt
        final_prompt = f"""
        [CONTEXT]
        {context_clauses}

        [QUERY]
        {request.user_query}
        """

        # 4. Call OpenAI
        response = openai_client.chat.completions.create(
            model="o3",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": final_prompt}
            ],
            response_format={"type": "json_object"}
        )

        json_response = response.choices[0].message.content

        # 5. Return response
        return {
            "success": True,
            "processing_time": f"{time.time() - start_time:.2f}s",
            "data": json_response
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to process query: {str(e)}"
        }
