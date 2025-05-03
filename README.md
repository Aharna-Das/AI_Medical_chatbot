# 🩺 Medical Assistant Chatbot
This is a Streamlit web application that provides a conversational AI interface for assisting users with medical queries. It integrates retrieval-augmented generation (RAG) using Hugging Face models, FAISS-based vector search, and structured follow-up recommendations. Additionally, it supports CSV-based prediction and generates professional medical assessment reports based on the conversation.

🚀 Features
✅ Conversational Chat Interface for medical Q&A

🧠 Retrieval-Augmented Generation (RAG) with HuggingFace Transformers

🔍 FAISS Vector Store for context-aware answers

💬 Follow-up Question Generation for better medical insight

📊 Prediction API Integration for file-based medical predictions

📝 Auto-generated Medical Report based on the conversation history

🧰 Requirements
Python 3.7+

Streamlit

Requests

Langchain & associated modules

Huggingface Transformers & Endpoint access

FAISS

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
📁 Project Structure
bash
Copy
Edit
mainchatbot.py            # Main Streamlit application
vectorstore/db_faiss/     # FAISS vector DB files (must exist before running)
.env                      # Contains HF_TOKEN
🔐 Environment Variables
Create a .env file with:

ini
Copy
Edit
HF_TOKEN=your_huggingface_token
Make sure your HuggingFace token has access to the endpoint model (mistralai/Mistral-7B-Instruct-v0.3).

▶️ Running the App
bash
Copy
Edit
streamlit run mainchatbot.py
Make sure your local prediction API is running at:

cpp
Copy
Edit
http://127.0.0.1:5000
📤 Optional: CSV Prediction Endpoint
The chatbot supports file uploads to a local Flask-based prediction API. You can integrate your own model at:

python
Copy
Edit
http://127.0.0.1:5000
Expected response: JSON with prediction results.

📄 Medical Report Generation
After a chat session, click Generate Report to receive a structured assessment:

Medical Assessment

Recommendations

Follow-ups

📌 Notes
Ensure the vectorstore/db_faiss directory and embedding model are prepared in advance.

Designed for educational/clinical assistant use only — not a substitute for professional medical advice.

