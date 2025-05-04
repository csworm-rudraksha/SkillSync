from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import pandas as pd
import os
import itertools
import configparser
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import uuid

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load credentials
config = configparser.ConfigParser()
config.read('llm.cfg')
api_key = config.get('DEFAULT', 'API_KEY')
project_id = config.get('DEFAULT', 'PROJECT_ID')
model_id = config.get('DEFAULT', 'MODEL_ID')

# Load employee data
employee_data = pd.read_csv('static/employees4.csv')

# Setup ChromaDB
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma"
))
collection_name = "employee_bios"
if collection_name in [c.name for c in chroma_client.list_collections()]:
    collection = chroma_client.get_collection(collection_name)
else:
    collection = chroma_client.create_collection(name=collection_name)

# Load model
embedder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# Vectorize bios
def vectorize_employees():
    collection.delete(where={})
    for idx, row in employee_data.iterrows():
        name = row['Name']
        bio = row['Bio']
        embedding = embedder.encode(bio).tolist()
        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[bio],
            metadatas=[{"name": name}]
        )

vectorize_employees()

class ProjectDescriptionRequest(BaseModel):
    description: str
    group_size: int = 2

def get_iam_token(api_key: str) -> str:
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {'grant_type': 'urn:ibm:params:oauth:grant-type:apikey', 'apikey': api_key.strip()}
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()
    return response.json()['access_token']

def extract_skills(description: str, access_token: str) -> List[str]:
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2025-02-11"
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
        'X-Project-Id': project_id
    }
    prompt = (
        "Extract only the key technical skills (technologies, tools, frameworks, languages) "
        "required from the following project description.\n"
        "Output only a clean comma-separated list of skills without numbering, bullet points, or explanations.\n\n"
        f"Project description:\n{description}"
    )
    payload = {
        "input": prompt,
        "model_id": model_id,
        "project_id": project_id
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    skills_text = result['results'][0]['generated_text'].strip()
    return [s.strip().lower() for s in skills_text.split(",") if s.strip()]

def match_employees(description: str) -> List[str]:
    description_embedding = embedder.encode(description).tolist()
    results = collection.query(query_embeddings=[description_embedding], n_results=10)
    matched_names = [metadata['name'] for metadata in results['metadatas'][0]]
    return matched_names

def get_similarity_score(text1: str, text2: str) -> float:
    emb1 = embedder.encode(text1)
    emb2 = embedder.encode(text2)
    return float(1 - (1 - (emb1 @ emb2) / (sum(x**2 for x in emb1)**0.5 * sum(x**2 for x in emb2)**0.5)))

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/employee", response_class=HTMLResponse)
async def employee_dashboard(request: Request):
    return templates.TemplateResponse("employee.html", {"request": request})

@app.get("/manager", response_class=HTMLResponse)
async def manager_dashboard(request: Request):
    return templates.TemplateResponse("manager.html", {"request": request})

@app.get("/suggested", response_class=HTMLResponse)
async def suggested_page(request: Request):
    return templates.TemplateResponse("suggested.html", {"request": request})

@app.get("/approved", response_class=HTMLResponse)
async def approved_page(request: Request):
    return templates.TemplateResponse("approved.html", {"request": request})

@app.post("/extract_skills")
def extract_project_skills(request: ProjectDescriptionRequest):
    try:
        access_token = get_iam_token(api_key)
        skills = extract_skills(request.description, access_token)
        matched_employees = match_employees(" ".join(skills))[:6]

        group_size = min(request.group_size, len(matched_employees))
        group_similarity_scores = []

        for group in itertools.combinations(matched_employees, group_size):
            similarities = []
            for emp1, emp2 in itertools.combinations(group, 2):
                bio1 = employee_data.loc[employee_data['Name'] == emp1, 'Bio'].values[0]
                bio2 = employee_data.loc[employee_data['Name'] == emp2, 'Bio'].values[0]
                similarity = get_similarity_score(bio1, bio2)
                similarities.append(similarity)
            avg_similarity = round(sum(similarities) / len(similarities), 2) if similarities else 0.0
            group_similarity_scores.append({
                "employees": list(group),
                "average_similarity": avg_similarity
            })

        group_similarity_scores.sort(key=lambda x: x["average_similarity"], reverse=True)
        top_groups = group_similarity_scores[:4]

        if group_size >= 4 and top_groups:
            worst_group = top_groups[-1]["employees"]
            pairwise_scores = []
            for emp1, emp2 in itertools.combinations(worst_group, 2):
                bio1 = employee_data.loc[employee_data['Name'] == emp1, 'Bio'].values[0]
                bio2 = employee_data.loc[employee_data['Name'] == emp2, 'Bio'].values[0]
                score = get_similarity_score(bio1, bio2)
                pairwise_scores.append((emp1, emp2, score))

            emp_score_map = {}
            for emp1, emp2, score in pairwise_scores:
                emp_score_map[emp1] = emp_score_map.get(emp1, 0) + score
                emp_score_map[emp2] = emp_score_map.get(emp2, 0) + score

            avg_scores = {emp: total / (group_size - 1) for emp, total in emp_score_map.items()}
            outlier_employee = min(avg_scores.items(), key=lambda x: x[1])[0]
            top_groups[-1]["outlier_employee"] = outlier_employee

        return {
            "extracted_skills": skills,
            "matched_employees": matched_employees,
            "group_similarity_scores": top_groups
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))