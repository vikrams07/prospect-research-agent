# ========================================
# 🏆 Hackathon Template Notebook
# Prospect Research Agent - Subtask 1
# ========================================

# --------- 1. INSTALL DEPENDENCIES ---------
# Installs necessary packages for scraping, parsing, and structured generation
!pip install -q beautifulsoup4 requests google-genai pydantic

import json
import re
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# For securely storing and retrieving the API key
from google.colab import userdata

# --------- 2. CONFIGURATION ---------
# 🔑 Retrieve your API key from Colab Secrets
API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=API_KEY)

# --------- 3. STRUCTURED DATA SCHEMA ---------
# Enforces the precise JSON schema format required by the judges
class CompanyProfile(BaseModel):
    website_name: str = Field(description="The clean public name of the website")
    company_name: str = Field(description="The official legal or corporate name of the company")
    address: str = Field(description="Physical address of the company. Put 'N/A' if not found.")
    mobile_number: str = Field(description="Contact phone number. Put 'N/A' if not found.")
    mail: list[str] = Field(description="List of email addresses found on the site. Return empty list if none.")
    core_service: str = Field(description="Detailed summary of what core service or product they provide.")
    target_customer: str = Field(description="The primary target market or ideal customer profile for this company.")
    probable_pain_point: str = Field(description="A highly specific business/operational bottleneck this company likely faces based on their scale and tech stack.")
    outreach_opener: str = Field(description="A personalized, compelling cold outreach intro line leveraging their services and addressing their pain point.")

# --------- 4. TOKEN OPTIMIZATION & CLEANING ---------
def clean_html(html_content: str) -> str:
    """Strips tracking scripts, css styles, navbars, and footers to protect context window tokens."""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Strip layout boilerplate elements
    for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        element.decompose()
        
    text = soup.get_text(separator=" ")
    # Flatten whitespace chunks
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:6000] # Cap input character string size to avoid bloating token overhead

# --------- 5. FUZZY MATCHING / SMART DEEP CRAWLING ---------
def extract_smart_links(base_url: str, html_content: str) -> list[str]:
    """Scans for high-value context pages using keyword matching."""
    soup = BeautifulSoup(html_content, "html.parser")
    target_keywords = re.compile(r'(about|contact|service|product|team)', re.IGNORECASE)
    discovered_urls = set()
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        text = a_tag.get_text()
        
        if target_keywords.search(href) or target_keywords.search(text):
            full_url = urljoin(base_url, href)
            # Check domain matching to stay within scope
            if urlparse(full_url).netloc == urlparse(base_url).netloc:
                discovered_urls.add(full_url)
                if len(discovered_urls) >= 3: # Limit secondary links to maximize rate-limiting defense
                    break
                    
    return list(discovered_urls)

# --------- 6. MULTI-APPROACH SCRAPING PIPELINE ---------
def scrape_website(url: str) -> str:
    """Executes a multi-page scraping sequence with anti-blocking headers."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        # Approach 1: Primary Target Index
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"Failed to load main page. Status code: {response.status_code}"
            
        main_html = response.text
        aggregated_text = f"--- MAIN PAGE ---\n{clean_html(main_html)}\n"
        
        # Approach 2: Smart Secondary Page Scrape
        sub_urls = extract_smart_links(url, main_html)
        for sub_url in sub_urls:
            try:
                sub_resp = requests.get(sub_url, headers=headers, timeout=5)
                if sub_resp.status_code == 200:
                    aggregated_text += f"\n--- SUB PAGE ({sub_url}) ---\n{clean_html(sub_resp.text)}\n"
            except Exception:
                continue # Skip broken nested links seamlessly
                
        return aggregated_text
        
    except Exception as e:
        return f"Error scraping site: {str(e)}"

# --------- 7. REQUIRED FUNCTION (DO NOT CHANGE STRUCTURE) ---------
def enrich_company(url: str) -> dict:
    """
    Input: Company URL
    Output: Structured company profile (STRICT SCHEMA FORMAT)
    """
    site_context = scrape_website(url)
    
    # Prompt guarding structure to block AI hallucinations
    prompt = f"""
    You are an expert B2B Sales Prospecting Agent. Analyze the following scraped text content from a company website:
    
    URL: {url}
    Website Content:
    {site_context}
    
    CRITICAL INSTRUCTIONS:
    1. Extract the actual data requested in the schema.
    2. DO NOT hallucinate, guess, or fabricate phone numbers, physical addresses, or emails. If they are not explicitly written in the provided content, use "N/A" (or an empty array for 'mail').
    3. Synthesize high-value insights for 'probable_pain_point' and 'outreach_opener' based on their true core services.
    """
    
    try:
        # Call Gemini utilizing deterministic response parameters and strict schema parsing
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CompanyProfile,
                temperature=0.1, # Keep close to 0 to eliminate randomness
            ),
        )
        return json.loads(response.text)
        
    except Exception as e:
        # Schema Stability Guard Fallback
        return {
            "website_name": urlparse(url).netloc,
            "company_name": "N/A",
            "address": "N/A",
            "mobile_number": "N/A",
            "mail": [],
            "core_service": f"Failed to analyze due to error: {str(e)}",
            "target_customer": "N/A",
            "probable_pain_point": "N/A",
            "outreach_opener": "N/A"
        }

# --------- 8. THE GOLDEN RULE (MAIN EXECUTION) ---------
if __name__ == "__main__":
    print("🚀 Golden Rule Activated: Awaiting judge test verification array input...")
    
    # Mandatory interactive prompt required by evaluation metrics
    raw_input = input("\n📥 Paste JSON array of URLs: ").strip()
    
    try:
        urls = json.loads(raw_input)
        if not isinstance(urls, list):
            raise ValueError("Input must be formatted as a JSON list array.")
    except Exception as e:
        print(f"❌ Invalid JSON format input! Falling back to testing defaults. Error: {e}")
        urls = ["https://example.com"]

    results = []
    for url in urls:
        if not url.startswith("http"):
            url = "https://" + url
            
        data = enrich_company(url)
        results.append(data)
            
    # Evaluation Step 3 Output Requirement: Generate results.json file
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Render final JSON output array directly inside the execution view
    print("\n=== FINAL OUTPUT ===\n")
    print(json.dumps(results, indent=2))
