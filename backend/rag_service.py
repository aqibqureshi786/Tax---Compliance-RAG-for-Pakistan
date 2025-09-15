"""
RAG Service for Tax Compliance system.
Handles question answering using retrieved documents and Google Gemini LLM.
"""

import logging
from typing import List, Dict, Any, Optional
import json

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document

from document_processor import TaxDocumentProcessor
from llm_providers import LLMProviderManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaxRAGService:
    """RAG service for tax compliance queries."""
    
    def __init__(self, provider: str = "auto", **provider_kwargs):
        """
        Initialize the RAG service.
        
        Args:
            provider: LLM provider ("auto", "gemini", "ollama", "openai")
            **provider_kwargs: Additional arguments for the provider
        """
        self.provider_name = provider
        self.provider_kwargs = provider_kwargs
        self.document_processor = TaxDocumentProcessor()
        self.llm_manager = LLMProviderManager()
        
        # Initialize LLM
        self.setup_llm()
        
        # Create custom prompt template
        self.setup_prompt_template()
    
    def setup_llm(self):
        """Initialize LLM based on provider choice."""
        try:
            if self.provider_name == "auto":
                # Use the best available provider
                self.llm = self.llm_manager.get_best_available_llm()
                logger.info("Initialized LLM with auto-selected provider")
            else:
                # Use specific provider
                provider = self.llm_manager.get_provider(self.provider_name, **self.provider_kwargs)
                self.llm = provider.get_llm()
                logger.info(f"Initialized LLM with {self.provider_name} provider")
                
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            # Show available providers and setup instructions
            providers = self.llm_manager.get_available_providers()
            logger.info(f"Available providers: {providers}")
            raise
    
    def setup_prompt_template(self):
        """Set up the prompt template for tax queries."""
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a Pakistani tax law expert assistant. Use the following context from official tax documents to answer the question accurately.

Context from Pakistan Tax Documents:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context from official Pakistan tax documents
2. If the context doesn't contain enough information, say "I don't have sufficient information in the provided documents to answer this question completely"
3. Cite specific sections or rules when possible
4. Be precise and factual
5. If asked about tax calculations, provide step-by-step explanations
6. Use simple, clear language

Answer:"""
        )
    
    def get_relevant_context(self, query: str, max_results: int = 5) -> str:
        """
        Retrieve relevant context for the query.
        
        Args:
            query: User query
            max_results: Maximum number of document chunks to retrieve
            
        Returns:
            Formatted context string
        """
        try:
            # Search for relevant documents
            results = self.document_processor.search_documents(query, n_results=max_results)
            
            if not results:
                return "No relevant documents found."
            
            # Format context
            context_parts = []
            for i, result in enumerate(results, 1):
                source = result['metadata'].get('source', 'Unknown')
                doc_type = result['metadata'].get('document_type', 'general')
                content = result['content']
                
                context_parts.append(f"[Document {i} - {source} ({doc_type})]\n{content}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return "Error retrieving relevant documents."
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a tax-related question using RAG.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing answer and metadata
        """
        try:
            # Get relevant context
            context = self.get_relevant_context(question)
            
            # Generate prompt
            prompt = self.prompt_template.format(context=context, question=question)
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            
            # Get source documents for reference
            source_docs = self.document_processor.search_documents(question, n_results=3)
            sources = [
                {
                    "source": doc['metadata'].get('source', 'Unknown'),
                    "type": doc['metadata'].get('document_type', 'general'),
                    "relevance_score": 1 - doc['distance']  # Convert distance to relevance
                }
                for doc in source_docs
            ]
            
            return {
                "answer": response,
                "sources": sources,
                "question": question,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "question": question,
                "status": "error"
            }
    
    def search_documents(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents related to a query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of relevant document chunks
        """
        try:
            results = self.document_processor.search_documents(query, n_results=max_results)
            
            # Format results for frontend
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result['content'],
                    "source": result['metadata'].get('source', 'Unknown'),
                    "document_type": result['metadata'].get('document_type', 'general'),
                    "relevance_score": 1 - result['distance'],
                    "preview": result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def calculate_tax(self, income: float, tax_year: str = "2024-25") -> Dict[str, Any]:
        """
        Calculate income tax based on current rates.
        Note: This is a simplified calculation. For accurate calculations,
        consult the actual tax documents or a tax professional.
        
        Args:
            income: Annual income in PKR
            tax_year: Tax year
            
        Returns:
            Tax calculation breakdown
        """
        try:
            # This is a simplified calculation based on general rates
            # In a real system, you'd retrieve current rates from documents
            
            # Basic tax slabs for individuals (simplified)
            tax_slabs = [
                {"min": 0, "max": 600000, "rate": 0},
                {"min": 600001, "max": 1200000, "rate": 0.025},
                {"min": 1200001, "max": 2200000, "rate": 0.125},
                {"min": 2200001, "max": 3200000, "rate": 0.20},
                {"min": 3200001, "max": 4100000, "rate": 0.25},
                {"min": 4100001, "max": float('inf'), "rate": 0.35}
            ]
            
            total_tax = 0
            tax_breakdown = []
            remaining_income = income
            
            for slab in tax_slabs:
                if remaining_income <= 0:
                    break
                
                taxable_in_slab = min(remaining_income, slab["max"] - slab["min"] + 1)
                if slab["min"] == 0:
                    taxable_in_slab = min(remaining_income, slab["max"])
                
                tax_in_slab = taxable_in_slab * slab["rate"]
                total_tax += tax_in_slab
                
                tax_breakdown.append({
                    "slab": f"PKR {slab['min']:,} - {slab['max'] if slab['max'] != float('inf') else 'Above'}{':,}' if slab['max'] != float('inf') else ''}",
                    "rate": f"{slab['rate'] * 100}%",
                    "taxable_amount": taxable_in_slab,
                    "tax_amount": tax_in_slab
                })
                
                remaining_income -= taxable_in_slab
            
            return {
                "income": income,
                "total_tax": total_tax,
                "net_income": income - total_tax,
                "effective_rate": (total_tax / income * 100) if income > 0 else 0,
                "tax_breakdown": tax_breakdown,
                "tax_year": tax_year,
                "note": "This is a simplified calculation. Please consult official tax documents or a tax professional for accurate calculations.",
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error calculating tax: {str(e)}")
            return {
                "error": f"Error calculating tax: {str(e)}",
                "status": "error"
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            doc_stats = self.document_processor.get_collection_stats()
            
            return {
                "document_count": doc_stats["total_documents"],
                "collection_name": doc_stats["collection_name"],
                "provider_name": self.provider_name,
                "llm_type": getattr(self.llm, '_llm_type', 'unknown'),
                "status": "operational"
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {
                "document_count": 0,
                "collection_name": "unknown",
                "provider_name": self.provider_name,
                "llm_type": "unknown",
                "status": "error"
            }

