# 🎓 AI Transcript Strategist 
<img width="1517" height="691" alt="image" src="https://github.com/user-attachments/assets/e2b6bf31-d857-4ec3-a3d3-0f2b49ca3a7f" />


An intelligent, multi-modal academic advising agent that automates transcript analysis, GPA calculation, and strategic course planning.

## 🚀 Architecture
This project is built on a distributed microservices architecture:
* **Frontend/OCR Engine:** Python, Gradio, Groq Llama-3.2-Vision (Extracts structured JSON from raw transcript images).
* **Orchestration Backend:** n8n (Handles data routing, logical branching, and API calls).
* **Calculation Engine:** Custom JavaScript module for 100% deterministic GPA math and "What-If" predictive scenarios.
* **LLM Advisor:** Groq API (Provides empathetic, contextual academic advice).

## ✨ Key Features
* **Vision OCR:** Automatically parses legacy transcript images into structured JSON data.
* **Predictive Analytics:** Generates "What-If" scenarios, calculating exact CGPA gains if specific weak courses are improved.
* **Agentic Branching:** Automatically escalates critical academic standings (CGPA < 2.0) to a different AI prompt to draft urgent advisor meeting emails.
* **Resilient Parsing:** Custom Python logic to parse messy, human-typed course data securely.

## 🛠️ Tech Stack
`Python` `Gradio` `n8n` `Groq API` `LLMs` `JavaScript` `REST Webhooks`
