# 🩺 Medical Assistant Chatbot

A conversational AI-based assistant for medical consultations. Built using Streamlit, Hugging Face Transformers, Langchain, FAISS, and integrates a file-based prediction system with automated medical report generation.

![image](https://github.com/user-attachments/assets/8332cdb5-1fff-469c-ab15-bf064786df26)

---

## 🚀 Features

- **Conversational Chat Interface**  
  Chat with an AI that understands medical queries and provides detailed responses.

- **Retrieval-Augmented Generation (RAG)**  
  Uses vector similarity search with FAISS to retrieve relevant medical context before answering.

- **Hugging Face LLM Integration**  
  Incorporates `mistralai/Mistral-7B-Instruct-v0.3` via Hugging Face Inference Endpoints for generating responses.

- **Follow-up Question Suggestions**  
  Dynamically generates follow-up medical questions to guide the consultation process.

- **Medical Report Generator**  
  Automatically generates a formal medical report at the end of the consultation.
  ![image](https://github.com/user-attachments/assets/e636f8bc-70de-43f8-bf69-7845909d6441)

---

## 🧰 Prerequisites

### Python Version
- Python 3.7 or higher

### Required Libraries

Make sure to install all required Python packages:

```bash
pip install -r requirements.txt
```
🔐 Environment Variables
Create a .env file in your project root with the following content:
```bash
HF_TOKEN=your_huggingface_access_token
Replace your_huggingface_access_token with your actual Hugging Face token (must have access to inference endpoints).
```
📁 Project Structure
```bash
.
├── mainchatbot.py                # Main Streamlit chatbot script
├── .env                          # Environment config for Hugging Face token
├── requirements.txt              # Python dependencies
└── vectorstore/
    └── db_faiss/                 # FAISS vector DB directory
```
📝 Note: Ensure the FAISS vector store (db_faiss) is initialized before starting the app.

▶️ Usage
Start the chatbot locally using Streamlit:

```bash
streamlit run mainchatbot.py
```
This will open a browser window at http://localhost:8501 with the chat interface.
