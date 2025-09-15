"""
FastAPI backend for Tax Compliance RAG system.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from rag_service import TaxRAGService
from document_processor import TaxDocumentProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Pakistan Tax Compliance RAG API",
    description="API for Pakistan Tax Compliance RAG system using LangChain and Google Gemini",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
rag_service: Optional[TaxRAGService] = None
document_processor: Optional[TaxDocumentProcessor] = None

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str = Field(..., description="Tax-related question to ask")
    max_sources: int = Field(default=5, description="Maximum number of source documents to include")

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    question: str
    timestamp: datetime
    status: str

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query for documents")
    max_results: int = Field(default=10, description="Maximum number of results to return")

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str
    total_results: int
    timestamp: datetime

class TaxCalculationRequest(BaseModel):
    income: float = Field(..., description="Annual income in PKR", gt=0)
    tax_year: str = Field(default="2024-25", description="Tax year")

class TaxCalculationResponse(BaseModel):
    income: float
    total_tax: float
    net_income: float
    effective_rate: float
    tax_breakdown: List[Dict[str, Any]]
    tax_year: str
    note: str
    timestamp: datetime
    status: str

class SystemStats(BaseModel):
    document_count: int
    collection_name: str
    provider_name: str
    llm_type: str
    status: str
    uptime: str
    timestamp: datetime

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global rag_service, document_processor
    
    try:
        logger.info("🚀 Starting Tax Compliance RAG API...")
        
        # Initialize document processor
        document_processor = TaxDocumentProcessor()
        logger.info("✅ Document processor initialized")
        
        # Initialize RAG service with auto provider selection
        rag_service = TaxRAGService(provider="auto")
        logger.info("✅ RAG service initialized")
        
        # Check if documents are processed
        stats = document_processor.get_collection_stats()
        if stats["total_documents"] == 0:
            logger.warning("⚠️  No documents found in vector store. Consider running document processing.")
        else:
            logger.info(f"📚 Found {stats['total_documents']} documents in vector store")
        
        logger.info("🎉 Tax Compliance RAG API started successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {str(e)}")
        raise

# Health check endpoint
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "message": "Pakistan Tax Compliance RAG API",
        "status": "operational",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    try:
        if rag_service is None:
            raise HTTPException(status_code=503, detail="RAG service not initialized")
        
        stats = rag_service.get_system_stats()
        return {
            "status": "healthy",
            "services": {
                "rag_service": "operational",
                "document_processor": "operational",
                "vector_store": "operational"
            },
            "stats": stats,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# Question answering endpoint
@app.post("/ask", response_model=QuestionResponse, tags=["Question Answering"])
async def ask_question(request: QuestionRequest):
    """
    Ask a tax-related question and get an AI-powered answer.
    """
    try:
        if rag_service is None:
            raise HTTPException(status_code=503, detail="RAG service not initialized")
        
        logger.info(f"Received question: {request.question}")
        
        # Add timeout handling
        import asyncio
        try:
            # Run with timeout (30 seconds)
            result = await asyncio.wait_for(
                asyncio.to_thread(rag_service.answer_question, request.question),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"Question timed out: {request.question}")
            return QuestionResponse(
                answer="Your question is taking too long to process. Please try asking a simpler question or break it into smaller parts.",
                sources=[],
                question=request.question,
                timestamp=datetime.now(),
                status="timeout"
            )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["answer"])
        
        return QuestionResponse(
            answer=result["answer"],
            sources=result["sources"][:request.max_sources],
            question=result["question"],
            timestamp=datetime.now(),
            status=result["status"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# Document search endpoint
@app.post("/search", response_model=SearchResponse, tags=["Document Search"])
async def search_documents(request: SearchRequest):
    """
    Search for relevant tax documents based on a query.
    """
    try:
        if rag_service is None:
            raise HTTPException(status_code=503, detail="RAG service not initialized")
        
        logger.info(f"Received search query: {request.query}")
        
        # Search documents
        results = rag_service.search_documents(request.query, max_results=request.max_results)
        
        return SearchResponse(
            results=results,
            query=request.query,
            total_results=len(results),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

# Tax calculation endpoint
@app.post("/calculate-tax", response_model=TaxCalculationResponse, tags=["Tax Calculation"])
async def calculate_tax(request: TaxCalculationRequest):
    """
    Calculate income tax based on provided income and tax year.
    """
    try:
        if rag_service is None:
            raise HTTPException(status_code=503, detail="RAG service not initialized")
        
        logger.info(f"Calculating tax for income: PKR {request.income:,}")
        
        # Calculate tax
        result = rag_service.calculate_tax(request.income, request.tax_year)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
        
        return TaxCalculationResponse(
            income=result["income"],
            total_tax=result["total_tax"],
            net_income=result["net_income"],
            effective_rate=result["effective_rate"],
            tax_breakdown=result["tax_breakdown"],
            tax_year=result["tax_year"],
            note=result["note"],
            timestamp=datetime.now(),
            status=result["status"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating tax: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating tax: {str(e)}")

# System statistics endpoint
@app.get("/stats", response_model=SystemStats, tags=["System"])
async def get_system_stats():
    """
    Get system statistics and status.
    """
    try:
        if rag_service is None:
            raise HTTPException(status_code=503, detail="RAG service not initialized")
        
        stats = rag_service.get_system_stats()
        
        return SystemStats(
            document_count=stats["document_count"],
            collection_name=stats["collection_name"],
            provider_name=stats["provider_name"],
            llm_type=stats["llm_type"],
            status=stats["status"],
            uptime="Running",  # Could implement actual uptime tracking
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting system stats: {str(e)}")

# Document processing endpoint (for admin use)
@app.post("/process-documents", tags=["Admin"])
async def process_documents(background_tasks: BackgroundTasks):
    """
    Process all documents in the Documents folder (admin endpoint).
    This runs in the background to avoid timeout issues.
    """
    try:
        if document_processor is None:
            raise HTTPException(status_code=503, detail="Document processor not initialized")
        
        # Add document processing to background tasks
        background_tasks.add_task(process_documents_background)
        
        return {
            "message": "Document processing started in background",
            "status": "processing",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error starting document processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting document processing: {str(e)}")

async def process_documents_background():
    """Background task for document processing."""
    try:
        global document_processor
        logger.info("🔄 Starting background document processing...")
        success = document_processor.process_all_documents()
        
        if success:
            logger.info("✅ Background document processing completed successfully")
        else:
            logger.error("❌ Background document processing failed")
    except Exception as e:
        logger.error(f"Error in background document processing: {str(e)}")

# Error handlers
from fastapi.responses import JSONResponse

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "The requested endpoint does not exist",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
