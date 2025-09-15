# 🏛️ Pakistan Tax Compliance RAG

AI-powered tax law assistant for Pakistan using official FBR documents and Google Gemini AI.

## ✨ Features

- **💬 Intelligent Q&A**: Ask complex questions about Pakistan's tax laws
- **🔍 Smart Document Search**: Search through official tax documents with AI
- **💰 Tax Calculator**: Calculate income tax with detailed breakdowns
- **⚡ Fast Responses**: Powered by Google Gemini AI with timeout handling
- **📚 Comprehensive Coverage**: Income Tax Ordinance 2001, Rules 2002, WHT rates

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Gemini API** (free):
   - Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Add to `backend/llm_providers.py` (line 89): `api_key="YOUR_KEY_HERE"`

3. **Start backend**:
   ```bash
   python backend/main.py
   ```

4. **Start frontend** (new terminal):
   ```bash
   streamlit run frontend/app.py
   ```

5. **Access**: Open http://localhost:8502 (or check terminal for exact port)

## Project Structure

```
├── Documents/              # Tax documents (PDFs)
├── backend/               # FastAPI backend
│   ├── main.py           # API server
│   ├── document_processor.py  # PDF processing
│   ├── rag_service.py    # Q&A logic
│   └── llm_providers.py  # LLM integrations
├── frontend/             # Streamlit frontend
│   └── app.py           # Web interface
├── chroma_db/           # Vector database (auto-created)
├── requirements.txt     # Dependencies
└── .env                # API keys
```

## Documents Included

- Income Tax Ordinance 2001 (Updated July 2025)
- Income Tax Rules 2002 (Updated November 2023)  
- WHT Rate Card

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Python web framework)
- **Backend**: FastAPI (Python API framework)
- **AI**: Google Gemini API (fast, free AI model)
- **Vector DB**: ChromaDB (local vector storage)
- **Processing**: LangChain (document processing & RAG)
- **Embeddings**: HuggingFace Sentence Transformers

## 📊 System Requirements

- **Python**: 3.8+
- **RAM**: 4GB+ recommended
- **Storage**: 2GB for dependencies + documents
- **Internet**: Required for Gemini API calls

## 🎯 Usage Examples

**Q&A Examples:**
- "What is the tax rate for income above PKR 1,000,000?"
- "How is foreign income taxed in Pakistan?"
- "What are the withholding tax rates for contractors?"

**Calculator:**
- Input your annual income to get detailed tax breakdown
- Supports different income types and deductions

---

**Disclaimer**: For informational purposes only. Consult tax professionals for official advice.