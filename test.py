import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch
from datetime import datetime
import json
import google.generativeai as genai
import os
from typing import List, Dict

class MedicalChatbotGemini:
    def __init__(self, google_api_key: str):
        # Initialize Gemini
        genai.configure(api_key=google_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Initialize the sentence transformer
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize conversation history and patient responses
        self.conversation_history = []
        self.patient_responses = {}
        
        # Generate initial questions using Gemini
        self.knowledge_base = self.generate_initial_questions()
        
        # Create FAISS index
        self.create_faiss_index()
        
    def generate_initial_questions(self) -> Dict:
        """Generate initial medical screening questions using Gemini"""
        prompt = """
        Generate a comprehensive list of medical screening questions in JSON format.
        Include general questions and specific follow-up questions for different symptoms.
        Format the response as a JSON object with two keys:
        1. 'questions': list of general screening questions
        2. 'follow_ups': dictionary mapping symptoms to relevant follow-up questions
        Keep the questions professional and medical in nature.
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Clean and parse the response
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            knowledge_base = json.loads(response_text)
            return knowledge_base
        except Exception as e:
            print(f"Error generating questions: {e}")
            # Fallback to default questions if Gemini fails
            return {
                "questions": [
                    "What are your current symptoms?",
                    "When did these symptoms begin?",
                    "Have you had any previous medical conditions?",
                    "Are you currently taking any medications?",
                    "Do you have any allergies?"
                ],
                "follow_ups": {
                    "pain": ["Where is the pain located?", "How severe is the pain?"],
                    "fever": ["How high is your temperature?", "When did the fever start?"]
                }
            }

    def create_faiss_index(self):
        """Create FAISS index for semantic search"""
        question_embeddings = self.sentence_transformer.encode(self.knowledge_base["questions"])
        dimension = question_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(question_embeddings).astype('float32'))

    def find_similar_questions(self, query: str, k: int = 2) -> List[str]:
        """Find similar questions using FAISS"""
        query_vector = self.sentence_transformer.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), k)
        return [self.knowledge_base["questions"][i] for i in indices[0]]

    def get_follow_up_questions(self, response: str) -> List[str]:
        """Get relevant follow-up questions based on user response"""
        follow_ups = []
        for keyword, questions in self.knowledge_base["follow_ups"].items():
            if keyword.lower() in response.lower():
                follow_ups.extend(questions)
        return follow_ups

    def generate_gemini_report(self) -> Dict:
        """Generate a medical report using Gemini based on conversation history"""
        # Prepare conversation summary for Gemini
        conversation_summary = "\n".join([
            f"Q: {interaction['bot']}\nA: {interaction['user']}"
            for interaction in self.conversation_history
            if 'bot' in interaction
        ])

        prompt = f"""
        Based on the following patient conversation, generate a detailed medical report.
        Include sections for summary, key symptoms, potential concerns, and recommendations.
        
        Conversation History:
        {conversation_summary}
        
        Format the report in JSON with the following structure:
        {{
            "summary": "Brief overview of the consultation",
            "key_symptoms": ["list", "of", "symptoms"],
            "potential_concerns": ["list", "of", "concerns"],
            "recommendations": ["list", "of", "recommendations"],
            "follow_up_needed": boolean,
            "urgency_level": "low/medium/high"
        }}
        """

        try:
            response = self.model.generate_content(prompt)
            report_text = response.text.strip()
            if report_text.startswith("```json"):
                report_text = report_text[7:-3]
            report = json.loads(report_text)
            
            # Add metadata
            report["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report["conversation_history"] = self.conversation_history
            
            return report
        except Exception as e:
            print(f"Error generating report: {e}")
            return self.generate_fallback_report()

    def generate_fallback_report(self) -> Dict:
        """Generate a basic report if Gemini fails"""
        return {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": "Medical Chat Session Report",
            "conversation_history": self.conversation_history,
            "key_symptoms": list(self.patient_responses.values()),
            "recommendations": ["Please consult with a healthcare provider for personalized medical advice"],
            "follow_up_needed": True,
            "urgency_level": "medium"
        }

    def chat(self):
        """Main chat loop"""
        print("Medical Chatbot: Hello! I'm here to help understand your symptoms. What brings you in today?")
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
                
            self.conversation_history.append({"user": user_input})
            
            # Find and ask relevant questions
            similar_questions = self.find_similar_questions(user_input)
            for question in similar_questions:
                print(f"Medical Chatbot: {question}")
                response = input("You: ")
                self.conversation_history.append({"bot": question, "user": response})
                self.patient_responses[question] = response
                
                # Ask follow-up questions
                follow_ups = self.get_follow_up_questions(response)
                for follow_up in follow_ups:
                    print(f"Medical Chatbot: {follow_up}")
                    follow_up_response = input("You: ")
                    self.conversation_history.append({"bot": follow_up, "user": follow_up_response})
                    self.patient_responses[follow_up] = follow_up_response
            
            print("Medical Chatbot: Is there anything else you'd like to tell me?")
        
        # Generate and display report using Gemini
        report = self.generate_gemini_report()
        print("\nMedical Chat Session Report:")
        print(json.dumps(report, indent=2))
        return report

# Example usage
if __name__ == "__main__":
    # Replace with your Google API key
    GOOGLE_API_KEY = "AIzaSyAWPLVVdieyMBaqa0t69wBeoyAMRMrg2wc"
    
    chatbot = MedicalChatbotGemini(GOOGLE_API_KEY)
    chatbot.chat()