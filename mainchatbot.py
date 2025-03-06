import os
import streamlit as st
import json
import requests
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
def call_prediction_api(data):
    url = "http://127.0.0.1:5000"  # Replace with your API's URL
    # Properly format files parameter as a dictionary with key-value pairs
    files = {'file': (data.name, data.getvalue(), 'text/csv')}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Prediction failed. Please try again later."}

def generate_report():
    report = "Medical Assistant Chatbot Report\n\n"
    for message in st.session_state.messages:
        if message['role'] == 'user':
            report += f"User: {message['content']}\n"
        elif message['role'] == 'assistant':
            report += f"Assistant: {message['content']}\n"

    print(report)
    REPORT_PROMPT_TEMPLATE = f"""
                            Context: {report}
                            Based on the provided medical report above, generate a structured and professional medical assessment without nameor anyother personal details . The report should include only the following sections:
                            
                            **Medical Assessment**:

                            Summarize the key findings from the report, including symptoms, possible diagnoses, and clinical observations.
                            Clearly state any identified conditions or potential concerns.
                            Recommendations:

                            Provide medical advice, treatment plans, or suggested medications based on the assessment.
                            Include any necessary lifestyle modifications.
                            Follow-ups:

                            Outline any required future consultations, tests, or specialist referrals.
                            Mention the recommended timeline for follow-up visits or additional diagnostics.
                            Ensure clarity, professionalism, and accuracy in summarizing the key details of the report while maintaining a medical tone."""
    return REPORT_PROMPT_TEMPLATE


def main():
    st.title("Medical Assistant Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'suggested_question' not in st.session_state:
        st.session_state.suggested_question = []


    # File upload button for CSV files
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        df = call_prediction_api(uploaded_file)
        st.write("Uploaded CSV file:")
        st.write(df)

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
   
    


    # Chat input
    prompt = st.chat_input("Ask a medical question...")
    if 'button_pressed' not in st.session_state:
        st.session_state.button_pressed = False
    
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        # Clear previous suggestions when a new question is asked
        #st.session_state.suggested_questions = ''
        
        CUSTOM_PROMPT_TEMPLATE = """
        You are a medical assistant chatbot. Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Don't provide anything out of the given context and don't mention about context.

        After your answer, generate a follow-up questions that would be helpful to gather more information or explore related medical topics.
        
        Format your response as a JSON object with two keys:
        1. "answer": Your answer to the user's question
        2. "follow_up_question":  A follow-up question for the user in next paragraph
        
        Context: {context}
        Question: {question}
        
        JSON response:
        """
        
        st.session_state.HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        st.session_state.HF_TOKEN = os.environ.get("HF_TOKEN")

        if not st.session_state.HF_TOKEN:
            st.error("HF_TOKEN environment variable is not set. Please set it and try again.")
            return

        try: 
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store. Please make sure the path exists.")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=st.session_state.HUGGINGFACE_REPO_ID, HF_TOKEN=st.session_state.HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            with st.spinner("Processing your question..."):
                response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            
        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")
            st.session_state.messages.append({'role': 'assistant', 'content': f"I'm sorry, I encountered an error: {str(e)}"})
        
#
        try:
            json_response = json.loads(result)
            answer = json_response.get("answer", "")
            follow_up_question = json_response.get("follow_up_question", "")
                
            st.session_state.suggested_question = follow_up_question
                    
            result_to_show = f"{answer}"

        except json.JSONDecodeError:
            # If JSON parsing fails, show the raw result
            result_to_show = result
                
        st.chat_message('assistant').markdown(result_to_show)
        st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})
        # Display the suggested follow-up question
        if st.session_state.suggested_question:
                st.write("**Follow-up Question:**")
                st.write(st.session_state.suggested_question) 
    
    if ' buttons_added ' not in st.session_state:
        st.session_state.buttons_added = False
    if st.session_state.messages and not st.session_state.buttons_added:
        if st.button("Generate Report"):
            report_prompt = generate_report()
                
         # Process the report with the LLM
            try:     
                report_llm = load_llm(huggingface_repo_id=st.session_state.HUGGINGFACE_REPO_ID, HF_TOKEN=st.session_state.HF_TOKEN)
                with st.spinner("Generating medical report..."):
                    report_response = report_llm.invoke(report_prompt)
                    
                st.session_state.generated_report = report_response
                st.chat_message('assistant').markdown(report_response)
                st.session_state.messages.append({'role': 'assistant', 'content': report_response})
            except Exception as e:
                st.error(f"Error generating the report: {str(e)}")
                st.session_state.generated_report = f"Report generation failed: {str(e)}"
        
                
                
        st.session_state.buttons_added = True
        # Force the app to rerun to show the suggested questions
        #st.rerun()

if __name__ == "__main__":
    main()