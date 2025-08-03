import os
from fastapi import FastAPI
from neo4j import GraphDatabase
import openai
from pydantic import BaseModel

# --- Configuration ---
# Best Practice: Load credentials from environment variables
AURA_URI = os.getenv("AURA_URI", "neo4j+s://6f9aa9c3.databases.neo4j.io")
AURA_USER = os.getenv("AURA_USER", "neo4j")
AURA_PASSWORD = os.getenv("AURA_PASSWORD","ZkQ5bYSkAHkajmXH-UnseaZqkKM2HB8c_EKOJEilWHs")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Initialize Clients ---
app = FastAPI()
neo4j_driver = GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PASSWORD))
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Pydantic model for request validation
class QueryRequest(BaseModel):
    user_query: str

# This is the system prompt you created earlier
SYSTEM_PROMPT = """
You are an AI policy assistant. Your only task is to answer a user's query directly and concisely based strictly on the provided context clauses.

Begin your answer with a clear "Yes" or "No". Then, in the same sentence, briefly state the reason. Do not explain your reasoning process or mention the clause numbers.

Please respond in JSON format with the following structure:
{
  "answer": "Yes/No, brief reason here"
  
}
"""

@app.post("/query")
async def process_query(request: QueryRequest):
    # STEP 1: RETRIEVE CONTEXT FROM NEO4J (Your existing logic)
    # This is a placeholder for your actual retrieval query
    context_clauses = "Clause 3.2: A waiting period of 24 months is applicable for joint replacement surgery..."

    # STEP 2: AUGMENT - Combine everything into a final prompt for the LLM
    final_prompt = f"""
    [CONTEXT]
    {context_clauses}

    [QUERY]
    {request.user_query}
    """

    # STEP 3: GENERATE - Call the OpenAI API
    try:
        response = openai_client.chat.completions.create(
            model="o3",  # Using O3 model as specified
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": final_prompt}
            ],
            response_format={"type": "json_object"} # Enforces JSON output
        )
        
        # Extract the JSON content from the response
        json_response = response.choices[0].message.content
        return json_response

    except Exception as e:
        return {"error": f"Failed to call OpenAI API: {e}"}