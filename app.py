import os
import json
import re
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load local .env file (for local development only)
load_dotenv()

app = FastAPI(title="Prospect Research Agent UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Securely retrieve API Key from Render Environment Variable
API_KEY = os.environ.get('GOOGLE_API_KEY')
if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in environment variables!")

client = genai.Client(api_key=API_KEY)
DB_FILE = "results.json"

if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w") as f:
        json.dump([], f)

# --- MODELS & PIPELINE ---
class EnrichRequest(BaseModel):
    url: str
    website_name: str

class CompanyProfile(BaseModel):
    website_name: str
    company_name: str
    address: str
    mobile_number: str
    mail: list[str]
    core_service: str
    target_customer: str
    probable_pain_point: str
    outreach_opener: str

def clean_html(html_content: str) -> str:
    if not html_content: return ""
    soup = BeautifulSoup(html_content, "html.parser")
    for element in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        element.decompose()
    text = re.sub(r'\s+', ' ', soup.get_text(separator=' ')).strip()
    return text[:7000]

def scrape_website(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        return f"--- MAIN PAGE ---\n{clean_html(response.text)}"
    except Exception as e:
        return f"Scraping error: {str(e)}"

@app.post("/enrich", response_model=CompanyProfile)
async def enrich_company_endpoint(payload: EnrichRequest):
    url = payload.url if payload.url.startswith("http") else f"https://{payload.url}"
    site_context = scrape_website(url)
    
    prompt = f"Analyze: {url}. Context: {site_context}. Extract data strictly. No hallucinations."
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CompanyProfile,
                temperature=0.1,
            ),
        )
        result_data = json.loads(response.text)
        result_data["website_name"] = payload.website_name
    except Exception as e:
        result_data = {"website_name": payload.website_name, "company_name": "N/A", "address": "N/A", 
                       "mobile_number": "N/A", "mail": [], "core_service": f"Error: {str(e)}", 
                       "target_customer": "N/A", "probable_pain_point": "N/A", "outreach_opener": "N/A"}
    
    with open(DB_FILE, "r+") as f:
        data = json.load(f)
        data.append(result_data)
        f.seek(0)
        json.dump(data, f, indent=2)
    return result_data

@app.get("/results")
async def get_all_results():
    with open(DB_FILE, "r") as f: return json.load(f)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    # Return your HTML content here (truncated for brevity)
    return "..."
