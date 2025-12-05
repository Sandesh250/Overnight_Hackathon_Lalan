import os
import json
import smtplib
from email.mime.text import MIMEText
from collections import defaultdict
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
from dotenv import load_dotenv


from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docx import Document
from pypdf import PdfReader

load_dotenv()

app = FastAPI(title="File Classifier + Email Router")


@app.get("/", response_class=HTMLResponse)
async def root():
    return "<h2>File Classifier API</h2><p>Use <a href='/docs'>/docs</a> to try the endpoints.</p>"

FAV = Path(__file__).parent / "favicon.ico"

@app.get("/favicon.ico")
async def favicon():
    if FAV.exists():
        return FileResponse(FAV)
    return "", 204


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USER)


llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=GROQ_API_KEY)


DEPARTMENTS = [
    "Principal",
    "Finance",
    "Vice Principal",
    "Dean Student Affairs",
    "Dean Academics",
    "HODs of all Departments",
    "Controller of Examination",
    "Management",
    "HR Department",
    "Library Department"
]

DEPT_EMAIL_MAP = {
    "Principal": "sandeshspatrot@gmail.com",
    "Finance": "sandeshpatrot@gmail.com",
    "Vice Principal": "lalanpradeep26@gmail.com",
    "Dean Student Affairs": "lalanp.ai24@rvce.edu.in",
    "Dean Academics": "rohanshreedhar19@gmail.com",
    "HODs of all Departments": "rohans.cs24@rvce.edu.in",
    "Controller of Examination": "sandeshspam@gmail.com",
    "Management": "sandeshspatrot.cs24@rvce.edu.in",
    "HR Department": "rishabrajeshn.ai24@rvce.edu.in",
    "Library Department": "rishab090506@gmail.com",
}


def extract_text_from_pdf(file_obj) -> str:
    reader = PdfReader(file_obj)
    parts = []
    for p in reader.pages:
        try:
            t = p.extract_text()
        except Exception:
            t = None
        if t:
            parts.append(t)
    return "\n".join(parts)

def extract_text_from_docx(file_obj) -> str:
    doc = Document(file_obj)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_file(uploaded_file: UploadFile) -> str:
    name = uploaded_file.filename.lower()
    f = uploaded_file.file
    if name.endswith(".pdf"):
        return extract_text_from_pdf(f)
    elif name.endswith(".docx"):
        return extract_text_from_docx(f)
    elif name.endswith(".txt"):
        raw = f.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore")
        return str(raw)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, DOCX or TXT.")



def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100,
        length_function=lambda t: len(t.split()),
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c and c.strip()]



segregator_prompt = ChatPromptTemplate.from_template("""
You are an expert extractor whose job is to split the GIVEN CHUNK into the portions that are
explicitly or implicitly addressed to each of the following departments:

Departments:
{departments}

Return a JSON mapping department -> text (only departments present). Return strictly valid JSON.

CHUNK:
{chunk}
""")
segregator_parser = JsonOutputParser()

def segregate_chunk_to_dept_texts(chunk_text_input: str) -> Dict[str,str]:
    chain = segregator_prompt | llm | segregator_parser
    try:
        resp = chain.invoke({"departments": "\n".join(DEPARTMENTS), "chunk": chunk_text_input})
    except Exception:
        return {}
    if isinstance(resp, dict):
        return resp
    if isinstance(resp, str):
        try:
            return json.loads(resp)
        except Exception:
            return {}
    return {}


classifier_prompt = ChatPromptTemplate.from_template("""
You are an expert document classifier.

Departments:
- Principal
- Finance
- Vice Principal
- Dean Student Affairs
- Dean Academics
- HODs of all Departments
- Controller of Examination
- Management
- HR Department
- Library Department

For the given CHUNK, classify it into ALL relevant departments.
A chunk may belong to MULTIPLE departments.

Return output strictly in JSON format:
{
  "chunk": "<text>",
  "departments": ["Department1", "Department2", ...]
}

CHUNK:
{chunk}
""")
classifier_parser = JsonOutputParser()

def classify_subchunk_with_groq(chunk_text_input: str) -> Dict[str,Any]:
    chain = classifier_prompt | llm | classifier_parser
    try:
        resp = chain.invoke({"chunk": chunk_text_input})
    except Exception:
        return {"chunk": chunk_text_input, "departments": []}
    if isinstance(resp, dict):
        resp.setdefault("chunk", chunk_text_input)
        resp.setdefault("departments", [])
        return resp
    if isinstance(resp, str):
        try:
            parsed = json.loads(resp)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {"chunk": chunk_text_input, "departments": []}


KEYWORD_DEPT_MAP = {
    "exam": ["Controller of Examination", "Dean Academics"],
    "result": ["Controller of Examination"],
    "timetable": ["Controller of Examination", "Dean Academics"],
    "student": ["Dean Student Affairs"],
    "complaint": ["Dean Student Affairs"],
    "library": ["Library Department"],
    "book": ["Library Department"],
    "salary": ["HR Department"],
    "leave": ["HR Department", "HODs of all Departments"],
    "invoice": ["Management", "HODs of all Departments"],
    "purchase": ["Management", "HODs of all Departments"],
    "meeting": ["Principal", "Management"],
    "appointment": ["HR Department", "Principal"],
    "admission": ["Dean Academics"],
    "faculty": ["HODs of all Departments", "Dean Academics"]
}

def keyword_classify_list(text: str) -> List[str]:
    lower = text.lower()
    found = set()
    for kw, depts in KEYWORD_DEPT_MAP.items():
        if kw in lower:
            for d in depts:
                if d in DEPARTMENTS:
                    found.add(d)
    return list(found)


combine_prompt = ChatPromptTemplate.from_template("""
Produce STRICT JSON: {"department":"...", "subject":"...", "body":"...", "action_items":["..."]}

Department: {department}
Chunks:
{chunks}
""")
combine_parser = JsonOutputParser()

def combine_chunks_for_department(department: str, chunk_texts: List[str]) -> Dict[str,Any]:
    joined = "\n---\n".join(chunk_texts)
    chain = combine_prompt | llm | combine_parser
    try:
        resp = chain.invoke({"department": department, "chunks": joined})
    except Exception:
        return {
            "department": department,
            "subject": f"Document items for {department}",
            "body": joined,
            "action_items": [],
            "message": joined
        }
    if isinstance(resp, dict):
        resp.setdefault("department", department)
        resp.setdefault("subject", f"Document items for {department}")
        resp.setdefault("body", "")
        resp.setdefault("action_items", [])
        resp.setdefault("message", resp.get("body", ""))
        return resp
    if isinstance(resp, str):
        try:
            parsed = json.loads(resp)
            parsed.setdefault("department", department)
            parsed.setdefault("subject", f"Document items for {department}")
            parsed.setdefault("body", "")
            parsed.setdefault("action_items", [])
            parsed.setdefault("message", parsed.get("body", ""))
            return parsed
        except Exception:
            return {
                "department": department,
                "subject": f"Document items for {department}",
                "body": joined,
                "action_items": [],
                "message": joined
            }
    return {
        "department": department,
        "subject": f"Document items for {department}",
        "body": joined,
        "action_items": [],
        "message": joined
    }


def send_email_via_smtp(to_email: str, subject: str, body: str) -> Dict[str,Any]:
    if not SMTP_SERVER or not SMTP_USER or not SMTP_PASS:
        return {"to": to_email, "status": "skipped", "error": "SMTP not configured (check SMTP_SERVER/SMTP_USER/SMTP_PASS)"}
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL or SMTP_USER
    msg["To"] = to_email
    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(FROM_EMAIL or SMTP_USER, [to_email], msg.as_string())
        return {"to": to_email, "status": "sent", "error": None}
    except Exception as e:
        return {"to": to_email, "status": "error", "error": str(e)}


@app.post("/classify")
async def classify_document(file: UploadFile):
 
    text = extract_text_from_file(file)
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No readable text found in file.")


    chunks = chunk_text(text)

  
    segregated_subchunks = []
    for c in chunks:
        dept_map = segregate_chunk_to_dept_texts(c)
        if isinstance(dept_map, dict) and len(dept_map) > 0:
            for dept, part_text in dept_map.items():
                if part_text and part_text.strip():
                    segregated_subchunks.append({"chunk": part_text.strip(), "dept_hint": dept})
        else:
            segregated_subchunks.append({"chunk": c.strip(), "dept_hint": None})

    results: List[Dict[str,Any]] = []
    for item in segregated_subchunks:
        sub = item["chunk"]
        resp = classify_subchunk_with_groq(sub)
        if not resp.get("departments"):
            
            
            kw = keyword_classify_list(sub)
            if kw:
                resp["departments"] = kw
            else:
                hint = item.get("dept_hint")
                if hint:
                    resp["departments"] = [hint]
                else:
                    resp["departments"] = []
        resp["_segregator_hint"] = item.get("dept_hint")
        results.append(resp)

    
    
    dept_to_chunks: Dict[str, List[str]] = defaultdict(list)
    for r in results:
        chunk_val = r.get("chunk", "")
        for dept in r.get("departments", []):
            if dept in DEPARTMENTS:
                dept_to_chunks[dept].append(chunk_val)


    if not dept_to_chunks:
        whole_depts = keyword_classify_list(text)
        if whole_depts:
            for d in whole_depts:
                dept_to_chunks[d].append(text)
        else:
          
          
            dept_to_chunks["Principal"].append(text)


    combined_messages: List[Dict[str,Any]] = []
    for dept, texts in dept_to_chunks.items():
        combined = combine_chunks_for_department(dept, texts)
        combined["message"] = combined.get("body", "")
        combined_messages.append(combined)

  
  
    emails_sent = []
    for msg in combined_messages:
        dept_name = msg.get("department")
        to_email = DEPT_EMAIL_MAP.get(dept_name)
        subject = msg.get("subject", f"Document items for {dept_name}")
        body = msg.get("body", msg.get("message", ""))
        if not to_email:
            emails_sent.append({
                "department": dept_name,
                "email": None,
                "status": "skipped",
                "error": "No predefined email for this department in DEPT_EMAIL_MAP"
            })
            continue
        send_result = send_email_via_smtp(to_email, subject, body)
        emails_sent.append({
            "department": dept_name,
            "email": to_email,
            "status": send_result.get("status"),
            "error": send_result.get("error")
        })

   
   
    return {
        "filename": file.filename,
        "segregated_output": results,
        "combined_messages": combined_messages,
        "emails_sent": emails_sent
    }
