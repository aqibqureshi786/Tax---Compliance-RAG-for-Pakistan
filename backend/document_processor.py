"""
Document processor for Tax Compliance RAG system.
Handles PDF processing, text extraction, and chunking for vector storage.
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaxDocumentProcessor:
    """Process tax documents and create vector embeddings for RAG system."""
    
    def __init__(self, documents_path: str = "Documents", chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            documents_path: Path to directory containing PDF documents
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
        """
        self.documents_path = Path(documents_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize ChromaDB
        self.setup_chromadb()
    
    def setup_chromadb(self):
        """Set up ChromaDB client and collection."""
        try:
            # Create ChromaDB client
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            
            # Create or get collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="tax_documents",
                metadata={"description": "Pakistan Tax Compliance Documents"}
            )
            
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of Document objects
        """
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source": pdf_path.name,
                    "file_type": "pdf",
                    "document_type": self._classify_document(pdf_path.name)
                })
            
            logger.info(f"Extracted {len(documents)} pages from {pdf_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return []
    
    def _classify_document(self, filename: str) -> str:
        """Classify document type based on filename."""
        filename_lower = filename.lower()
        
        if "income-tax-ordinance" in filename_lower:
            return "ordinance"
        elif "income-tax-rules" in filename_lower or "incometaxrules" in filename_lower:
            return "rules"
        elif "wht" in filename_lower or "withholding" in filename_lower:
            return "wht_rates"
        else:
            return "general"
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        try:
            chunked_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
            return chunked_docs
        except Exception as e:
            logger.error(f"Error chunking documents: {str(e)}")
            return []
    
    def store_embeddings(self, documents: List[Document]) -> bool:
        """
        Store document embeddings in ChromaDB.
        
        Args:
            documents: List of Document objects to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare data for ChromaDB
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            ids = [f"{doc.metadata.get('source', 'unknown')}_{i}" for i, doc in enumerate(documents)]
            
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Stored {len(documents)} document embeddings in ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            return False
    
    def process_all_documents(self) -> bool:
        """
        Process all PDF documents in the documents directory.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.documents_path.exists():
                logger.error(f"Documents directory {self.documents_path} does not exist")
                return False
            
            # Get all PDF files
            pdf_files = list(self.documents_path.glob("*.pdf"))
            
            if not pdf_files:
                logger.warning("No PDF files found in documents directory")
                return False
            
            logger.info(f"Found {len(pdf_files)} PDF files to process")
            
            all_documents = []
            
            # Process each PDF
            for pdf_file in pdf_files:
                logger.info(f"Processing {pdf_file.name}...")
                documents = self.extract_text_from_pdf(pdf_file)
                if documents:
                    all_documents.extend(documents)
            
            if not all_documents:
                logger.error("No documents were successfully processed")
                return False
            
            # Chunk documents
            chunked_documents = self.chunk_documents(all_documents)
            
            if not chunked_documents:
                logger.error("Failed to chunk documents")
                return False
            
            # Store embeddings
            success = self.store_embeddings(chunked_documents)
            
            if success:
                logger.info("All documents processed and stored successfully")
                return True
            else:
                logger.error("Failed to store embeddings")
                return False
                
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return False
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents based on query.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": "tax_documents"
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"total_documents": 0, "collection_name": "tax_documents"}

