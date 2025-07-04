import base64
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Config ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
api_key = os.getenv("OPENAI_API_KEY")
api_key1=os.getenv("OPENAI_API_KEY")

OUTPUT_DIR = Path(".") / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MARKDOWN_FILE = OUTPUT_DIR / "output7.md"
HTML_FILE = OUTPUT_DIR / "output7.html"

# === OpenRouter Client ===
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)
client1 = OpenAI(api_key=api_key1)


# === Input Schema ===
class BusinessName(BaseModel):
    user_prompt: str


class BusinessLogo(BaseModel):
    business_name: str
    style: str


class BusinessArtifacts(BaseModel):
    business_name: str
    user_prompt: str
    setup_type: str
    primary_location: str
    service_areas: list[str] = []
    timezone: str


# Business Artifacts preparation
# === Prompt Template ===
RESEARCH_TEMPLATE = """
You are Deep-Research-AI assistant, an expert assistant  in “Business Planning & Strategy” mode.Ensure that you have to give the maximum content you can give.
### 1 · Environment
• Enabled tools: Search Codebase, Web, Edit Files, Run Commands, MCP, Perplexity Search  
• Automation flags: Auto-fix Errors = ON | Auto-Apply Edits = OFF | Auto-Run = OFF  
• Citation policy: Embed inline citations after every statement drawn from external data.  
• Quality gate: Block output if required data or ROI calc is missing.
### 2 · Objective
Conduct full-stack research on a proposed business idea and deliver a single Markdown document that Claude can transform into an interactive artifact.  
### 3 · Required Sections & Checks  
1. Idea Analysis  
   a. Validate real pain-point (user-problem fit) within 5 lines.   
   b. List top 3 direct competitors with 1-line positioning each.This information should be get in the realtime only 
2. Market Research  
   a. TAM & SAM sizing (show method + numeric ranges)  
   b. Five hottest geographic markets with justification  
3. Financial Planning  
   a. Unit-economics table (CAC, LTV, gross margin %) which includes (| Metric | Value | Description |).You have to give the information in a single valued cards
   b. 12-month burn-rate forecast table (best / base / worst) which includes (| Month | Scenario | Revenue | COGS (35%) | Fixed + Mktg | Monthly Net | Cumulative Net |) based on the example investments regarding the business usecase.
   c. ROI Calculator: breakeven month & IRR based on base case .Create a Realtime ROI Calculator in UI with the Java Script and HTML which user can do and automate the calculations in UI.
4. Business Model Canvas Snapshot  
   a. Revenue streams (primary stream+ secondary stream) in different aspects. 
   b. Key activities & critical resources (maximum 6 bullet point pairs)
5. Risk & Mitigation(max 6 bullet pairs) including the some 5 liner context for each risk and mitigation.
6. Next-Step Roadmap(“Day 0 → MVP → Month 12” milestones) in the form of phases with the risks and mitigations which allows to think about growth,profits and losses also.
### 4 · Data-Gathering Workflow
1. Use Perplexity Search only for very broad desk research and academic/market reports.  
2. Pull structured datasets or APIs via MCP when numeric evidence is required.  
3. Validate code snippets or math in Run Commands; auto-fix on errors.  
4. Summarize findings in-place; never expose raw credentials or private keys.
### 5 · Output Format
```markdown
# [Proposed Business Name]
## 1. Idea Analysis
…
## 2. Market Research
…
## 3. Financial Planning
…
## 4. Business Model Canvas Snapshot
…
## 5. Risk & Mitigation
…
## 6. Next-Step Roadmap
…

Everything should be very clear and with the realtime information.Give the markdown file clearly showing every information

"""


# === Helpers ===
def openrouter_chat(model: str, system_prompt: str, user_prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            extra_body={
                "web_search_options": {
                    "search_context_size": "high"
                }
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        print("[OpenRouter SDK ERROR]", e)
        raise HTTPException(status_code=500, detail=str(e))


# === Core Logic ===
def generate_md_report(user_prompt: str) -> str:
    if not user_prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is empty")
    prompt = RESEARCH_TEMPLATE + user_prompt
    return openrouter_chat(
        model="perplexity/sonar-deep-research",
        system_prompt="You are a realtime Deep-Research AI assistant which can do research for the user prompt through various sources and will  extract  realtime information.You have to give the maximum information that you can.Every topic should have realtime description and follow some uniqueness.You have to give the realtime plots where you need to compare the things in the format of html and matplotlib.",
        user_prompt=prompt
    )


def generate_html_from_md(markdown: str) -> str:
    prompt = f"Transform the following markdown into an HTML layout using styled cards:\n\n{markdown}"
    return openrouter_chat(
        model="anthropic/claude-3.7-sonnet",
        system_prompt=f"""
        You are an HTML transformer that turns markdown business reports into styled card layouts.",
        You have to use the fixed template for every response.It should be very attractive and use the designs like cards etc whatever you need to make the presentation so beautiful to watch and include all the information in a proper manner.
        you have to use the nice background colors also instead of just plain white background for the attractive things.Choose the suitable background themes mostly.Frequently you can take the light themes also based on the user prompt.
        You can generate the  high quality graphs and all required for the easily analysis of the statistics based on the markdown file.
        you have to use the navigation bar like navbar which will be very attractive for the easily navigation of the sections.
        """,
        user_prompt=prompt
    )


# === Artifact Endpoint ===
@app.post("/generate-artifact")
async def generate_artifact(data: BusinessArtifacts):
    prompt = f"""
            I want to create a business report based on the following parameters:
            - Business Name: {data.business_name}
            - Vision/Problem Statement: {data.user_prompt}
            - Business Setup Type: {data.setup_type}
            - Primary Location: {data.primary_location}
            - Service Areas: {', '.join(data.service_areas)}
            - Timezone: {data.timezone}
                """
    markdown = generate_md_report(prompt)
    html = generate_html_from_md(markdown)

    MARKDOWN_FILE.write_text(markdown, encoding="utf-8")
    HTML_FILE.write_text(html, encoding="utf-8")

    return {
        "markdown": markdown,
        "html": html,
        "markdown_file": str(MARKDOWN_FILE),
        "html_file": str(HTML_FILE)
    }


# Creating an unique name
def generate_unique_name(idea: str) -> str:
    return openrouter_chat(
        model="anthropic/claude-3.7-sonnet",
        system_prompt=f"You're an assistant that generates creative brand names based on the {idea}.The brand names should be very creative and vary based on the location.Do not repeat the name once it is generated,Generate the names dynamically for each request. ",
        user_prompt=f"Generate 1 unique, brandable business name ideas for this concept: {idea}. Just give the names only."
    )


@app.post("/generate-name")
async def generate_name(data: BusinessName):
    name_suggestion = generate_unique_name(data.user_prompt)
    return {"name": name_suggestion}


# Creating an unique logo
# === Style Prompt Map ===
STYLE_PROMPTS = {
    "Modern": "sleek, geometric font, minimal look",
    "Playful": "rounded font, vibrant colors, friendly tone",
    "Professional": "classic font, navy blue tones, corporate style",
    "Creative": "artistic lettering, abstract brushstroke vibe",
    "Tech": "futuristic typography, digital theme",
    "Elegant": "serif font, gold on white, luxury branding"
}


# === Helpers ===
def generate_unique_logo_url(name: str, style: str) -> str:
    style_desc = STYLE_PROMPTS.get(style, "clean and modern")
    prompt = (
        f"Logo with only the text '{name}', in {style_desc}. "
        f"Create a clean, centered logo that displays only the business name."
        f"Use a bold, modern sans-serif font in dark gray or black. The logo must have a pure white background, "
        f"no icon, no illustrations, no decorations, and no tagline. Do not include any symbols, images, or colors other than the text. "
        f"Flat vector style. Center the text in the image."
        f"For every request you can generate the logo dyanamically.Donot repeat the logo once generated."
    )
    try:
        response = client1.images.generate(
            model="dall-e-2",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        return response.data[0].url
    except Exception as e:
        print("[OpenRouter SDK ERROR - Logo Gen]", e)
        raise HTTPException(status_code=500, detail=str(e))


def fetch_image_as_base64(image_url: str) -> str:
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")
    except Exception as e:
        print("[Image Fetch ERROR]", e)
        raise HTTPException(status_code=500, detail="Unable to fetch or encode image.")


# === API Endpoint ===
@app.post("/generate-logo")
async def generate_logo(data: BusinessLogo):
    if not data.business_name or not data.style:
        raise HTTPException(status_code=400, detail="Both business_name and style are required.")
    url = generate_unique_logo_url(data.business_name, data.style)
    base64_image = fetch_image_as_base64(url)
    return {"base64_logo": base64_image}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
