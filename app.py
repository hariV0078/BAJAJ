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
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load env variables (if using a .env file)
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI()

# CORS (allow frontend access if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Models
class UserQuery(BaseModel):
    query: str

# Cache to avoid recomputation
@lru_cache()
def embed_text(text: str) -> np.ndarray:
    return np.random.rand(768)

async def extract_text_from_pdf(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail="Failed to fetch PDF")

            data = await response.read()
            doc = fitz.open(stream=data, filetype="pdf")
            full_text = "\n".join(page.get_text() for page in doc)
            return full_text

@app.get("/")
async def root():
    return {"status": "Server is running."}

@app.post("/hackrx/run")
async def run_query(query: UserQuery, authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid token")

    token = authorization[7:]  # Skip "Bearer "
    if token != os.getenv("AUTH_TOKEN"):
        raise HTTPException(status_code=403, detail="Unauthorized")

    logger.info(f"Received query: {query.query}")

    # Simulate embedding and vector matching
    query_embedding = embed_text(query.query)
    fake_document_embedding = embed_text("Relevant content here.")

    similarity_score = cosine_similarity([query_embedding], [fake_document_embedding])[0][0]

    # Fake response
    response = {
        "query": query.query,
        "matched_info": "Document about legal rights",
        "similarity": round(float(similarity_score), 4)
    }

    return response
