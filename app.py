from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional
import httpx
import asyncio
import os
import json
import logging
from datetime import datetime
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import openai
import fitz  # PyMuPDF
from docx import Document
import tempfile
import uvicorn
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models and indices
embedding_model = None
faiss_index = None
document_chunks = []
document_metadata = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and resources on startup"""
    global embedding_model, faiss_index
    
    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    logger.info("Initializing FAISS index...")
    # Initialize with dimension of the embedding model
    faiss_index = faiss.IndexFlatIP(384)  # all-MiniLM-L6-v2 has 384 dimensions
    
    logger.info("Application startup complete")
    yield
    logger.info("Application shutdown")

app = FastAPI(
    title="LLM-Powered Document Query System",
    description="Intelligent query-retrieval system for insurance, legal, HR, and compliance documents",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "e0e272cd2f3ac51a8dda3c63908707c12fc63da7d51f0c7a8fbba91e80db3a88")

# OpenAI Configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.warning("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

# Pydantic models
class DocumentQuery(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class ProcessingMetrics(BaseModel):
    processing_time: float
    token_usage: int
    chunks_processed: int
    similarity_scores: List[float]

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials

class DocumentProcessor:
    """Handles document parsing and text extraction"""
    
    @staticmethod
    async def download_document(url: str) -> bytes:
        """Download document from URL"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.content
        except Exception as e:
            logger.error(f"Error downloading document: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")

    @staticmethod
    def extract_text_from_pdf(content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(content)
                temp_file.flush()
                
                doc = fitz.open(temp_file.name)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                os.unlink(temp_file.name)
                return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise HTTPException(status_code=500, detail="Failed to extract text from PDF")

    @staticmethod
    def extract_text_from_docx(content: bytes) -> str:
        """Extract text from DOCX content"""
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(content)
                temp_file.flush()
                
                doc = Document(temp_file.name)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                os.unlink(temp_file.name)
                return text
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            raise HTTPException(status_code=500, detail="Failed to extract text from DOCX")

    @classmethod
    async def process_document(cls, url: str) -> str:
        """Process document and extract text"""
        content = await cls.download_document(url)
        
        # Determine file type from URL or content
        url_lower = str(url).lower()
        if url_lower.endswith('.pdf') or b'%PDF' in content[:10]:
            return cls.extract_text_from_pdf(content)
        elif url_lower.endswith('.docx') or b'PK' in content[:2]:
            return cls.extract_text_from_docx(content)
        else:
            # Try to decode as text
            try:
                return content.decode('utf-8')
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="Unsupported file format")

class TextChunker:
    """Handles text chunking for embeddings"""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence.split())
            
            if current_size + sentence_size > chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'text': current_chunk.strip(),
                    'start_sentence': max(0, i - len(current_chunk.split('. '))),
                    'end_sentence': i,
                    'word_count': current_size
                })
                
                # Start new chunk with overlap
                overlap_sentences = '. '.join(current_chunk.split('. ')[-overlap//10:])
                current_chunk = overlap_sentences + '. ' + sentence if overlap_sentences else sentence
                current_size = len(current_chunk.split())
            else:
                current_chunk += '. ' + sentence if current_chunk else sentence
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'start_sentence': max(0, len(sentences) - len(current_chunk.split('. '))),
                'end_sentence': len(sentences),
                'word_count': current_size
            })
        
        return chunks

class EmbeddingSearch:
    """Handles embedding generation and similarity search"""
    
    @staticmethod
    def create_embeddings(texts: List[str]) -> np.ndarray:
        """Create embeddings for text chunks"""
        global embedding_model
        if embedding_model is None:
            raise HTTPException(status_code=500, detail="Embedding model not initialized")
        
        embeddings = embedding_model.encode(texts, convert_to_tensor=False)
        return np.array(embeddings).astype('float32')
    
    @staticmethod
    def build_faiss_index(embeddings: np.ndarray) -> None:
        """Build FAISS index from embeddings"""
        global faiss_index
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        faiss_index.reset()
        faiss_index.add(embeddings)
    
    @staticmethod
    def search_similar(query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar text chunks"""
        global embedding_model, faiss_index, document_chunks
        
        if embedding_model is None or faiss_index is None:
            raise HTTPException(status_code=500, detail="Search index not initialized")
        
        # Create query embedding
        query_embedding = embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = faiss_index.search(query_embedding, min(k, len(document_chunks)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(document_chunks):
                results.append({
                    'chunk': document_chunks[idx],
                    'similarity_score': float(score),
                    'index': int(idx)
                })
        
        return results

class LLMProcessor:
    """Handles LLM queries and response generation"""
    
    @staticmethod
    async def generate_answer(question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer using OpenAI GPT"""
        if not openai.api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        # Prepare context
        context = "\n\n".join([
            f"Context {i+1} (Score: {chunk['similarity_score']:.3f}):\n{chunk['chunk']['text']}"
            for i, chunk in enumerate(context_chunks[:3])  # Use top 3 chunks
        ])
        
        # Create prompt
        prompt = f"""Based on the following context from insurance policy documents, please answer the question accurately and concisely.

Context:
{context}

Question: {question}

Instructions:
1. Answer based only on the provided context
2. If the information is not available in the context, state "Information not available in the provided context"
3. Provide specific details like waiting periods, coverage limits, conditions, etc. when available
4. Be precise and factual

Answer:"""

        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert insurance policy analyzer. Provide accurate, concise answers based on policy documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip()
            token_usage = response.usage.total_tokens
            
            return {
                'answer': answer,
                'token_usage': token_usage,
                'context_used': len(context_chunks),
                'similarity_scores': [chunk['similarity_score'] for chunk in context_chunks]
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")

# API Routes
@app.get("/")
async def root():
    return {"message": "LLM-Powered Document Query System", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query_system(
    request: DocumentQuery,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
) -> QueryResponse:
    """Main endpoint for document querying"""
    global document_chunks, document_metadata
    
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"Processing document: {request.documents}")
        
        # Step 1: Download and extract text
        document_text = await DocumentProcessor.process_document(str(request.documents))
        logger.info(f"Extracted {len(document_text)} characters from document")
        
        # Step 2: Chunk text
        document_chunks = TextChunker.chunk_text(document_text)
        logger.info(f"Created {len(document_chunks)} text chunks")
        
        # Step 3: Create embeddings and build index
        chunk_texts = [chunk['text'] for chunk in document_chunks]
        embeddings = EmbeddingSearch.create_embeddings(chunk_texts)
        EmbeddingSearch.build_faiss_index(embeddings)
        logger.info("Built search index")
        
        # Step 4: Process questions
        answers = []
        total_tokens = 0
        
        for question in request.questions:
            logger.info(f"Processing question: {question[:50]}...")
            
            # Search for relevant chunks
            similar_chunks = EmbeddingSearch.search_similar(question, k=5)
            
            # Generate answer
            result = await LLMProcessor.generate_answer(question, similar_chunks)
            answers.append(result['answer'])
            total_tokens += result['token_usage']
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Completed processing in {processing_time:.2f}s, used {total_tokens} tokens")
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hackrx/metrics")
async def get_processing_metrics(
    request: DocumentQuery,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
) -> ProcessingMetrics:
    """Get detailed processing metrics"""
    start_time = datetime.utcnow()
    
    # Simulate processing (in real implementation, this would be actual processing)
    await asyncio.sleep(0.1)  # Simulate processing time
    
    processing_time = (datetime.utcnow() - start_time).total_seconds()
    
    return ProcessingMetrics(
        processing_time=processing_time,
        token_usage=150,  # Placeholder
        chunks_processed=len(document_chunks) if document_chunks else 0,
        similarity_scores=[0.85, 0.72, 0.65, 0.58, 0.45]  # Placeholder
    )

# For Gunicorn deployment, we don't need the uvicorn.run() call
# The app instance will be imported directly by Gunicorn

if __name__ == "__main__":
    # This is only for local development
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
