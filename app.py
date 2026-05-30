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
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>B2B Prospect Research Agent</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            .hidden { display: none; }
        </style>
    </head>
    <body class="bg-slate-900 text-slate-100 font-sans min-h-screen p-8">
        <div class="max-w-4xl mx-auto">
            <h1 class="text-4xl font-bold text-teal-400 mb-6">🎯 AI Prospect Research Agent</h1>
            
            <div class="bg-slate-800 p-6 rounded-lg border border-slate-700 shadow-lg">
                <input id="webName" placeholder="Company Name" class="w-full bg-slate-900 p-3 mb-2 rounded border border-slate-600 text-white">
                <input id="webUrl" placeholder="https://example.com" class="w-full bg-slate-900 p-3 mb-4 rounded border border-slate-600 text-white">
                <button onclick="enrichTarget()" class="bg-teal-600 px-6 py-2 rounded text-white font-bold hover:bg-teal-500">
                    Run Enrichment
                </button>
            </div>
            
            <div id="statusIndicator" class="hidden mt-4 text-teal-400">Processing...</div>
            <div id="singleOutput" class="mt-6 space-y-4"></div>
        </div>

        <script>
            async function enrichTarget() {
                const url = document.getElementById("webUrl").value;
                const name = document.getElementById("webName").value;
                document.getElementById("statusIndicator").classList.remove("hidden");
                
                try {
                    const res = await fetch("/enrich", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({url: url, website_name: name})
                    });
                    const data = await res.json();
                    document.getElementById("singleOutput").innerHTML = `
                        <div class="bg-slate-800 p-4 rounded border border-teal-500">
                            <h3 class="font-bold text-teal-400">${data.website_name}</h3>
                            <p>${data.core_service}</p>
                            <p class="text-slate-400 italic mt-2">"${data.outreach_opener}"</p>
                        </div>`;
                } catch(e) {
                    alert("Error: " + e);
                } finally {
                    document.getElementById("statusIndicator").classList.add("hidden");
                }
            }
        </script>
    </body>
    </html>
    """

    
