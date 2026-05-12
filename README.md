---
title: Chat with PDF
---

# 💬 Chat with PDF — AI Document Q&A

> Upload any PDF. Ask anything. Get instant answers with source references.
<!-- ![Demo GIF](assets/demo.gif) -->

## ✨ Features
- 📄 Upload any PDF document (reports, contracts, research papers)
- 🤖 Powered by Google Gemini 2.5 Flash — completely free
- 🧠 Remembers conversation context across multiple questions
- 📍 Cites which parts of the document were used to answer
- ⚡ Real-time typewriter response animation
- 🌐 Fully deployed — no installation needed

## 🚀 Live Demo
**[Try it here →](https://chat-with-pdf-mwzpf2odf8voxdd8h3gb52.streamlit.app/)**

## 🛠️ Tech Stack
| Layer | Technology |
|-------|-----------|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Google text-embedding-004 |
| Vector DB | FAISS (local) |
| RAG Framework | LangChain |
| PDF Parsing | pdfplumber |
| UI | Streamlit |
| Deployment | Streamlit Cloud |

## 📁 Project Structure
```
chat-with-pdf/
├── app/
│   ├── loader.py        # PDF text extraction
│   ├── chunker.py       # Text splitting (500 tokens, 50 overlap)
│   ├── vectorstore.py   # FAISS embedding + retrieval
│   └── qa_chain.py      # RAG chain + memory
├── assets/
│   ├── styles.css       # Glassmorphism UI styles
│   └── demo.gif         # Demo recording
├── app.py               # Streamlit entry point
└── requirements.txt
```

## ⚙️ Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/chat-with-pdf
cd chat-with-pdf
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "GOOGLE_API_KEY=your_key_here" > .env
streamlit run app.py
```
Get a free API key at: [aistudio.google.com](https://aistudio.google.com)

## 💡 Use Cases
- **Legal teams** — query contracts and agreements instantly
- **Researchers** — extract insights from papers without reading in full
- **HR departments** — search through policy documents and handbooks
- **Students** — study smarter by chatting with textbooks

## 📬 Built by
Yash Chavda — B.Tech CSE, Karnavati University
Intern @ IIT Gandhinagar (Makers Bhavan)
[LinkedIn](http://linkedin.com/in/yashchavda) · [GitHub](https://github.com/Yashchavdafr)
---
