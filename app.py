from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
from datetime import datetime
import logging
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2
import docx
import re
import json
import openai
from nltk.tokenize import sent_tokenize

# Initialize FastAPI app
app = FastAPI(
    title="Intelligent Query-Retrieval System",
    description="LLM-powered system for processing documents and answering queries",
    version="1.0.0"
)

# Security
security = HTTPBearer()
TEAM_TOKEN = "e0e272cd2f3ac51a8dda3c63908707c12fc63da7d51f0c7a8fbba91e80db3a88"

# Models
class DocumentInput(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index.index"
CHUNK_SIZE = 500  # characters
OVERLAP = 50      # characters
MAX_TOKENS = 4096  # For GPT-4
TEMPERATURE = 0.3

# Initialize components
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
faiss_index = None
document_chunks = []
metadata = []

# Utility Functions
def download_file(url: str) -> str:
    """Download document from URL"""
    local_path = f"temp_{datetime.now().timestamp()}"
    if url.endswith('.pdf'):
        local_path += '.pdf'
    elif url.endswith('.docx'):
        local_path += '.docx'
    else:
        raise ValueError("Unsupported file format")
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_path

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF"""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from DOCX"""
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def preprocess_text(text: str) -> str:
    """Clean and normalize text"""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace
    text = text.strip()
    return text

def chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - OVERLAP
    return chunks

def create_faiss_index(embeddings: np.ndarray):
    """Create and save FAISS index"""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    return index

def load_or_create_index(chunks: List[str]):
    """Load or create FAISS index"""
    global faiss_index, document_chunks, metadata
    
    if os.path.exists(FAISS_INDEX_PATH):
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        embeddings = embedding_model.encode(chunks)
        faiss_index = create_faiss_index(embeddings)
    
    document_chunks = chunks
    metadata = [{"source": "document", "chunk_id": i} for i in range(len(chunks))]

def semantic_search(query: str, k: int = 3) -> List[dict]:
    """Perform semantic search using FAISS"""
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, k)
    
    results = []
    for i, distance in zip(indices[0], distances[0]):
        if i == -1:  # Invalid index
            continue
        results.append({
            "chunk": document_chunks[i],
            "metadata": metadata[i],
            "score": float(1 / (1 + distance))  # Convert distance to similarity score
        })
    
    return sorted(results, key=lambda x: x['score'], reverse=True)

def generate_response_with_llm(query: str, context: str) -> str:
    """Generate response using LLM with context"""
    prompt = f"""
    You are an intelligent query answering system for insurance, legal, HR, and compliance documents.
    Based on the following document context, answer the user's question precisely.
    If the information is not in the document, say "The document does not specify."

    Document Context:
    {context}

    Question: {query}
    Answer:"""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS // 2
    )
    
    return response.choices[0].message.content.strip()

def process_document(document_url: str):
    """Process document and create search index"""
    try:
        local_path = download_file(document_url)
        
        if document_url.endswith('.pdf'):
            text = extract_text_from_pdf(local_path)
        elif document_url.endswith('.docx'):
            text = extract_text_from_docx(local_path)
        else:
            raise ValueError("Unsupported file format")
        
        text = preprocess_text(text)
        chunks = chunk_text(text)
        load_or_create_index(chunks)
        
        os.remove(local_path)
        return True
    except Exception as e:
        logging.error(f"Error processing document: {str(e)}")
        return False

# API Endpoints
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_query(
    input_data: DocumentInput,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Process document and answer questions"""
    # Authentication
    if credentials.credentials != TEAM_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Process document
    if not process_document(input_data.documents):
        raise HTTPException(status_code=400, detail="Document processing failed")
    
    # Answer questions
    answers = []
    for question in input_data.questions:
        try:
            # Semantic search for relevant chunks
            search_results = semantic_search(question)
            context = "\n\n".join([res['chunk'] for res in search_results[:3]])
            
            # Generate answer with LLM
            answer = generate_response_with_llm(question, context)
            answers.append(answer)
        except Exception as e:
            answers.append(f"Error processing question: {str(e)}")
    
    return {"answers": answers}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
