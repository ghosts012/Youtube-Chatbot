# 📺 YouTube Video Chatbot (GenAI)

An interactive AI-powered application that allows users to chat with any YouTube video. Using **Gemini 1.5 Flash**, **LangChain**, and **FAISS**, the app extracts transcripts, indexes them into a vector store, and provides precise answers based *only* on the video content.

## 🚀 Features
* **Transcript Extraction:** Automatically fetches transcripts (English/Hindi) using `youtube-transcript-api`.
* **RAG Implementation:** Uses Retrieval-Augmented Generation to ensure the AI doesn't hallucinate and only answers from the video context.
* **Smart Memory:** Maintains chat history during your session using Streamlit's `session_state`.
* **Fast Search:** Uses FAISS (Facebook AI Similarity Search) for near-instant retrieval of relevant video segments.

## 🛠️ Tech Stack
- **Frontend:** [Streamlit](https://streamlit.io/)
- **LLM:** [Google Gemini 1.5 Flash](https://ai.google.dev/)
- **Orchestration:** [LangChain](https://www.langchain.com/)
- **Vector DB:** [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)

## 📋 Prerequisites
- Python 3.9 or higher
- A Google AI (Gemini) API Key

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/youtube-chatbot.git](https://github.com/ghosts012/youtube-chatbot.git)
   cd youtube-chatbot