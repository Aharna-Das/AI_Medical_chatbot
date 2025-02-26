import os
import streamlit as st
import json

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": 512
        }
    )
    return llm

def main():
    st.title("Medical Assistant Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'suggested_questions' not in st.session_state:
        st.session_state.suggested_questions = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    
    # Display suggested follow-up questions as buttons
    if st.session_state.suggested_questions:
        st.write("Suggested follow-up questions:")
        cols = st.columns(len(st.session_state.suggested_questions))
        for i, question in enumerate(st.session_state.suggested_questions):
            if cols[i].button(question):
                # If a suggested question is clicked, set it as the prompt
                prompt = question
                st.session_state.suggested_questions = []  # Clear suggestions
                st.rerun()  # Rerun to process the selected question
    
    # Chat input
    prompt = st.chat_input("Ask a medical question...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        # Clear previous suggestions when a new question is asked
        st.session_state.suggested_questions = []
        
        CUSTOM_PROMPT_TEMPLATE = """
        You are a medical assistant chatbot. Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Don't provide anything out of the given context.

        After your answer, generate 3 relevant follow-up questions that would be helpful to gather more information or explore related medical topics.
        
        Format your response as a JSON object with two keys:
        1. "answer": Your answer to the user's question
        2. "follow_up_questions": An array of 3 suggested follow-up questions
        
        Context: {context}
        Question: {question}
        
        JSON response:
        """
        
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        if not HF_TOKEN:
            st.error("HF_TOKEN environment variable is not set. Please set it and try again.")
            return

        try: 
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store. Please make sure the path exists.")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            with st.spinner("Processing your question..."):
                response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_documents = response["source_documents"]
            
            # Process the JSON response
            try:
                json_response = json.loads(result)
                answer = json_response.get("answer", "")
                follow_up_questions = json_response.get("follow_up_questions", [])
                
                # Save suggested questions for next interaction
                st.session_state.suggested_questions = follow_up_questions
                
                # Format source documents
                sources_text = "\n\n**Sources:**\n"
                for i, doc in enumerate(source_documents):
                    sources_text += f"{i+1}. {doc.metadata.get('source', 'Unknown')}\n"
                
                result_to_show = f"{answer}\n{sources_text}"
                
            except json.JSONDecodeError:
                # If JSON parsing fails, show the raw result
                result_to_show = f"{result}\n\n**Sources:**\n"
                for i, doc in enumerate(source_documents):
                    result_to_show += f"{i+1}. {doc.metadata.get('source', 'Unknown')}\n"
            
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})
            
            # Force the app to rerun to show the suggested questions
            st.rerun()

        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")
            st.session_state.messages.append({'role': 'assistant', 'content': f"I'm sorry, I encountered an error: {str(e)}"})

if __name__ == "__main__":
    main()