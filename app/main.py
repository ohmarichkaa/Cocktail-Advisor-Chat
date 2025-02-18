import pandas as pd
import openai
import numpy as np
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from types import SimpleNamespace
import faiss
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import re

DATASET_PATH = "data/cocktails.csv"

load_dotenv("set.env")

# Check API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found!")

dim = 768
db = faiss.IndexFlatL2(dim)
docstore = InMemoryDocstore()
index_to_docstore_id = SimpleNamespace()
vector_store = FAISS(
    db,
    OpenAIEmbeddings(openai_api_key=api_key),
    docstore,
    index_to_docstore_id
)

print("Vector database initialized successfully!")

user_preferences = []


def load_dataset():
    try:
        df = pd.read_csv(DATASET_PATH)
        if df.empty:
            raise ValueError("❌ The dataset is empty.")
        print("Dataset loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


# 🔹 Function 2: Index the dataset
def index_dataset(df):
    try:
        texts = []
        metadata_list = []
        for _, row in df.iterrows():
            name = row.get("name", "Unknown")
            ingredients = row.get("ingredients", "")
            category = row.get("category", "Uncategorized")
            text = f"Cocktail: {name}. Ingredients: {ingredients}. Category: {category}."
            texts.append(text)
            metadata_list.append({"name": name, "category": category})

        global vector_store
        vector_store = FAISS.from_texts(
            texts,
            OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
            metadatas=metadata_list
        )
        print("Dataset indexed into the vector store!")
    except Exception as e:
        print(f"Error indexing dataset: {e}")


def search_context(query: str):
    try:
        results = vector_store.similarity_search(query, k=5)
        context = "\n".join([res.page_content for res in results])
        return context
    except Exception as e:
        print(f"Error searching context: {e}")
        return ""


client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
def save_user_preferences(user_input: str):
    patterns = [
        r"my favorite ingredients are (.+)",
        r"i like (.+)",
        r"i love (.+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, user_input.lower())
        if match:
            ingredients = match.group(1).strip()
            vector_store.add_texts([f"[USER_PROVIDED] User likes: {ingredients}"])
            print(f"Saved favorite ingredients: {ingredients}")


def get_favorite_ingredients():
    try:
        results = vector_store.similarity_search("User likes:", k=5)
        favorites = [res.page_content.replace("User likes: ", "").replace("[USER_PROVIDED] ", "") for res in results]

        user_defined = [f for f in favorites if "[USER_PROVIDED]" in f]
        if user_defined:
            return ", ".join(set(user_defined)) + " (provided directly by user)"
        elif favorites:
            return ", ".join(set(favorites)) + " (inferred from cocktail analysis)"
        return "No favorite ingredients found."
    except Exception as e:
        print(f"Error retrieving favorite ingredients: {e}")
        return "Failed to retrieve favorites."


def generate_response(query: str, context: str):
    try:
        favorites = get_favorite_ingredients()

        if "provided directly by user" in favorites:
            context_with_preferences = f"User's favorite ingredients: {favorites}. Do not analyze cocktails; these ingredients were provided directly by the user."
        else:
            context_with_preferences = f"{context}\nUser's favorite ingredients: {favorites}"

        messages = [
            {"role": "system", "content": "You are a cocktail expert. Always respond in English. Text should not be bold.If the user provides ingredients directly, use only these ingredients and do not analyze cocktails."},
            {"role": "user", "content": f"Context: {context_with_preferences}\n\nQuestion: {query}"}
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=1,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error generating response: {e}")
        return "Failed to get a response from the model."


app = FastAPI()
templates = Jinja2Templates(directory="templates")


class UserInput(BaseModel):
    user_input: str

@app.on_event("startup")
def startup_event():
    df = load_dataset()
    if df is not None:
        index_dataset(df)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(input_data: UserInput):
    user_input = input_data.user_input    
    save_user_preferences(user_input)
    context = search_context(user_input)
    response = generate_response(user_input, context)
    return JSONResponse(content={"response": response})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
