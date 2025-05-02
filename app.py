from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import requests
import configparser
import pandas as pd
import itertools

app = FastAPI()

# CORS middleware for frontend API calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and Templates Setup
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

class ProjectDescriptionRequest(BaseModel):
    description: str
    group_size: int = 2

def get_iam_token(api_key: str) -> str:
    api_key = api_key.strip()
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {'grant_type': 'urn:ibm:params:oauth:grant-type:apikey', 'apikey': api_key}
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()
    return response.json()['access_token']

def extract_skills(description: str, access_token: str) -> str:
    url = f"https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2025-02-11"
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
    return skills_text

def match_skills_to_employees(skills_list):
    employee_matches = []
    for _, row in employee_data.iterrows():
        if int(row.get('Billability Hours', 1)) != 0:
            continue
        employee_skills = str(row['Skill']).split(',')
        employee_skills = [skill.strip().lower() for skill in employee_skills]
        match_count = sum(1 for skill in skills_list if skill.strip().lower() in employee_skills)
        employee_matches.append((row['Name'], match_count))
    employee_matches = sorted(employee_matches, key=lambda x: x[1], reverse=True)
    matching_employees = [name for name, count in employee_matches if count > 0]
    return matching_employees

def get_similarity_score(text1: str, text2: str, access_token: str) -> float:
    url = f"https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2025-02-11"
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
        'X-Project-Id': project_id
    }
    prompt = f"Calculate the semantic similarity score (0 to 1) between the following two texts:\nText 1: {text1}\nText 2: {text2}\nReturn only the numeric score."
    payload = {
        "input": prompt,
        "model_id": model_id,
        "project_id": project_id
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    generated_text = result['results'][0]['generated_text'].strip()
    try:
        similarity_score = float(generated_text)
    except ValueError:
        similarity_score = 0.0
    return similarity_score

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/employee", response_class=HTMLResponse)
async def employee_dashboard(request: Request):
    return templates.TemplateResponse("employee.html", {"request": request})

@app.get("/manager", response_class=HTMLResponse)
async def manager_dashboard(request: Request):
    return templates.TemplateResponse("manager.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

@app.post("/extract_skills")
def extract_project_skills(request: ProjectDescriptionRequest):
    try:
        access_token = get_iam_token(api_key)
        skills_text = extract_skills(request.description, access_token)
        skills_list = [skill.strip() for skill in skills_text.split(',') if skill.strip()]
        matched_employees = match_skills_to_employees(skills_list)[:6]

        group_size = min(request.group_size, len(matched_employees))
        group_similarity_scores = []

        for group in itertools.combinations(matched_employees, group_size):
            pairwise = list(itertools.combinations(group, 2))
            similarities = []
            for emp1, emp2 in pairwise:
                bio1 = employee_data.loc[employee_data['Name'] == emp1, 'Bio'].values[0]
                bio2 = employee_data.loc[employee_data['Name'] == emp2, 'Bio'].values[0]
                similarity = get_similarity_score(bio1, bio2, access_token)
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
                score = get_similarity_score(bio1, bio2, access_token)
                pairwise_scores.append((emp1, emp2, score))

            emp_score_map = {}
            for emp1, emp2, score in pairwise_scores:
                emp_score_map[emp1] = emp_score_map.get(emp1, 0) + score
                emp_score_map[emp2] = emp_score_map.get(emp2, 0) + score

            avg_scores = {emp: total / (group_size - 1) for emp, total in emp_score_map.items()}
            outlier_employee = min(avg_scores.items(), key=lambda x: x[1])[0]
            top_groups[-1]["outlier_employee"] = outlier_employee

        return {
            "extracted_skills": skills_list,
            "matched_employees": matched_employees,
            "group_similarity_scores": top_groups
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
