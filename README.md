# Cocktail Advisor Chat
## Overview

A Python-based chatbot using OpenAI's LLM and FAISS with Retrieval-Augmented Generation (RAG) to provide cocktail recommendations based on user queries and stored preferences.

## Setup Instructions
1. Clone the Repository:
```bash   
git clone https://github.com/ohmarichkaa/Cocktail-Advisor-Chat.git  
cd Cocktail-Advisor-Chat  
```

2. Install Dependencies:
```bash
pip install -r requirements.txt  
```

3. Configure Environment Variables:  
Create a `set.env` file with the following content:
```bash
OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXX  
```

5. Run the Application:
```bash
uvicorn app.main:app --reload  
```

Open in a browser: http://127.0.0.1:8000/

## Project Structure

project_root/  
    ├── app/  
    │   └── main.py  
    ├── templates/  
    │   └── index.html  
    ├── data/  
    │   └── cocktails.csv  
    ├── .gitignore  
    ├── set.env  
    ├── requirements.txt  
    └── README.md  

## Features
- Cocktail recommendations using a cocktail dataset and user-provided preferences.  
- User memory with FAISS for personalized advice.  
- English-only responses.  

## Example Usage
- User: "My favorite ingredients are mint and orange"  
- User: "What are my favorite ingredients?"  
- Bot: "Your favorite ingredients are mint and orange."  

- User: "Recommend cocktails with lemon"  
- Bot: "Here are 5 cocktails containing lemon: ..."  
