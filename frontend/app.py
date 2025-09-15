"""
Streamlit frontend for Pakistan Tax Compliance RAG system.
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, Any, List
import time

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"

# Page configuration
st.set_page_config(
    page_title="Pakistan Tax Compliance RAG",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f4e79;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    
    .source-box {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .tax-result {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-message {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 5px;
        padding: 1rem;
        color: #d32f2f;
    }
    
    .example-questions {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #2196f3;
    }
    
    .stButton > button {
        border-radius: 20px !important;
        border: 1px solid #e0e0e0 !important;
        background-color: white !important;
        color: #333 !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
        margin: 0.2rem 0 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #e3f2fd !important;
        border-color: #2196f3 !important;
        color: #1976d2 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health() -> bool:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def ask_question(question: str, max_sources: int = 5) -> Dict[str, Any]:
    """Send question to API and get response."""
    try:
        payload = {
            "question": question,
            "max_sources": max_sources
        }
        response = requests.post(f"{API_BASE_URL}/ask", json=payload, timeout=45)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try asking a simpler question or break your question into smaller parts."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to the backend API. Please ensure the backend is running."}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def search_documents(query: str, max_results: int = 10) -> Dict[str, Any]:
    """Search documents via API."""
    try:
        payload = {
            "query": query,
            "max_results": max_results
        }
        response = requests.post(f"{API_BASE_URL}/search", json=payload, timeout=15)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def calculate_tax(income: float, tax_year: str = "2024-25") -> Dict[str, Any]:
    """Calculate tax via API."""
    try:
        payload = {
            "income": income,
            "tax_year": tax_year
        }
        response = requests.post(f"{API_BASE_URL}/calculate-tax", json=payload, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def get_system_stats() -> Dict[str, Any]:
    """Get system statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def display_sources(sources: List[Dict[str, Any]]):
    """Display source documents."""
    if sources:
        st.markdown("### 📚 Sources")
        for i, source in enumerate(sources, 1):
            with st.expander(f"Source {i}: {source.get('source', 'Unknown')} ({source.get('type', 'general')})"):
                st.markdown(f"**Relevance Score:** {source.get('relevance_score', 0):.3f}")
                st.markdown(f"**Document Type:** {source.get('type', 'general')}")

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🏛️ Pakistan Tax Compliance RAG</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Tax Law Assistant for Pakistan</p>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("🚨 **API Server Not Running**")
        st.markdown("""
        The backend API is not accessible. Please ensure:
        1. The FastAPI server is running (`python backend/main.py`)
        2. Gemini API key is configured
        3. Internet connection is available for Gemini API
        
        **To start the backend:**
        ```bash
        cd backend
        python main.py
        ```
        """)
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🛠️ System Status")
        
        # Get system stats
        stats = get_system_stats()
        if "error" not in stats:
            st.success("✅ System Operational")
            st.metric("Documents Loaded", stats.get("document_count", 0))
            st.metric("LLM Model", stats.get("model_name", "Unknown"))
        else:
            st.error("❌ System Error")
            st.error(stats["error"])
        
        st.markdown("---")
        st.markdown("## 📋 Features")
        st.markdown("""
        - **Q&A:** Ask tax-related questions
        - **Document Search:** Find relevant tax documents  
        - **Tax Calculator:** Calculate income tax
        """)
        
        st.markdown("---")
        st.markdown("## ℹ️ About")
        st.markdown("""
        This RAG system uses:
        - **LangChain** for document processing
        - **ChromaDB** for vector storage
        - **Google Gemini** for cloud LLM
        - **FastAPI** for backend
        - **Streamlit** for frontend
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "🔍 Search Documents", "💰 Tax Calculator"])
    
    # Tab 1: Question Answering
    with tab1:
        st.markdown("## 💬 Ask Tax-Related Questions")
        st.markdown("Ask any question about Pakistan's tax laws, regulations, or procedures.")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Example questions
        st.markdown('<div class="example-questions">', unsafe_allow_html=True)
        st.markdown("### 💡 Try These Example Questions:")
        st.markdown("*Click any question below to auto-fill and test the system*")
        
        example_questions = [
            "What are the income tax rates for individuals in Pakistan?",
            "How is foreign income taxed in Pakistan?",
            "What are the withholding tax rates for contractors?",
            "What is the tax treatment of salary income?",
            "What are the filing requirements for income tax returns?",
            "How to calculate tax on rental income?",
            "What deductions are allowed for salaried individuals?",
            "What is the tax on capital gains?"
        ]
        
        # Display example questions in columns
        cols = st.columns(2)
        for i, example in enumerate(example_questions):
            col_idx = i % 2
            with cols[col_idx]:
                if st.button(f"📝 {example}", key=f"example_{i}", use_container_width=True):
                    st.session_state.selected_question = example
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Question input
        question = st.text_area(
            "Your Question:",
            value=st.session_state.get('selected_question', ''),
            placeholder="e.g., What are the current income tax rates for individuals in Pakistan?",
            height=100,
            key="question_input"
        )
        
        # Clear the selected question after it's been loaded
        if 'selected_question' in st.session_state:
            del st.session_state.selected_question
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🤖 Ask", type="primary")
        with col2:
            max_sources = st.slider("Max Sources", 1, 10, 5)
        
        if ask_button and question.strip():
            with st.spinner("🤔 Thinking... This may take a moment."):
                result = ask_question(question.strip(), max_sources)
            
            # Add to chat history
            st.session_state.chat_history.append({
                "question": question.strip(),
                "result": result,
                "timestamp": datetime.now()
            })
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("## 💭 Conversation History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {chat["question"]}</div>', unsafe_allow_html=True)
                
                if "error" in chat["result"]:
                    st.markdown(f'<div class="error-message"><strong>Error:</strong> {chat["result"]["error"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {chat["result"]["answer"]}</div>', unsafe_allow_html=True)
                    display_sources(chat["result"].get("sources", []))
                
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown("---")
        
        # Clear chat history
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
    
    # Tab 2: Document Search
    with tab2:
        st.markdown("## 🔍 Search Tax Documents")
        st.markdown("Search through Pakistan's official tax documents for specific information.")
        
        search_query = st.text_input(
            "Search Query:",
            placeholder="e.g., withholding tax rates, income tax exemptions"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            search_button = st.button("🔍 Search", type="primary")
        with col2:
            max_results = st.slider("Max Results", 5, 20, 10)
        
        if search_button and search_query.strip():
            with st.spinner("🔍 Searching documents..."):
                search_results = search_documents(search_query.strip(), max_results)
            
            if "error" in search_results:
                st.error(f"Search Error: {search_results['error']}")
            else:
                st.success(f"Found {search_results['total_results']} results for '{search_results['query']}'")
                
                for i, result in enumerate(search_results["results"], 1):
                    with st.expander(f"Result {i}: {result['source']} (Score: {result['relevance_score']:.3f})"):
                        st.markdown(f"**Document Type:** {result['document_type']}")
                        st.markdown(f"**Relevance Score:** {result['relevance_score']:.3f}")
                        st.markdown("**Content Preview:**")
                        st.markdown(result['preview'])
    
    # Tab 3: Tax Calculator
    with tab3:
        st.markdown("## 💰 Income Tax Calculator")
        st.markdown("Calculate your income tax based on Pakistan's current tax slabs.")
        
        st.warning("⚠️ **Disclaimer:** This is a simplified calculator for estimation purposes only. Please consult a tax professional or official FBR resources for accurate calculations.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            income = st.number_input(
                "Annual Income (PKR):",
                min_value=0.0,
                value=1000000.0,
                step=50000.0,
                format="%.0f"
            )
        
        with col2:
            tax_year = st.selectbox(
                "Tax Year:",
                ["2024-25", "2023-24", "2022-23"]
            )
        
        if st.button("💰 Calculate Tax", type="primary"):
            if income > 0:
                with st.spinner("🧮 Calculating tax..."):
                    calc_result = calculate_tax(income, tax_year)
                
                if "error" in calc_result:
                    st.error(f"Calculation Error: {calc_result['error']}")
                else:
                    # Display results
                    st.markdown('<div class="tax-result">', unsafe_allow_html=True)
                    st.markdown(f"## 📊 Tax Calculation Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Annual Income",
                            f"PKR {calc_result['income']:,.0f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Total Tax",
                            f"PKR {calc_result['total_tax']:,.0f}",
                            f"{calc_result['effective_rate']:.2f}% effective rate"
                        )
                    
                    with col3:
                        st.metric(
                            "Net Income",
                            f"PKR {calc_result['net_income']:,.0f}"
                        )
                    
                    # Tax breakdown
                    st.markdown("### 📋 Tax Breakdown by Slabs")
                    breakdown_data = []
                    for slab in calc_result['tax_breakdown']:
                        if slab['tax_amount'] > 0:
                            breakdown_data.append({
                                "Tax Slab": slab['slab'],
                                "Rate": slab['rate'],
                                "Taxable Amount": f"PKR {slab['taxable_amount']:,.0f}",
                                "Tax Amount": f"PKR {slab['tax_amount']:,.0f}"
                            })
                    
                    if breakdown_data:
                        st.table(breakdown_data)
                    
                    st.info(calc_result['note'])
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Please enter a valid income amount.")

if __name__ == "__main__":
    main()
