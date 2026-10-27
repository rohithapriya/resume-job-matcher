"""FastAPI service: upload a PDF resume and a job description, get a fit report back."""
import os
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.matcher import DEFAULT_MODEL, EmbeddingEncoder, analyze
from app.text_utils import extract_pdf_text

MAX_RESUME_BYTES = 5 * 1024 * 1024
MIN_JOB_CHARS = 30


class RequirementMatchOut(BaseModel):
    requirement: str
    best_resume_line: str
    similarity: float


class AnalyzeResponse(BaseModel):
    fit_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    matches: list[RequirementMatchOut]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup. Loading it per request would add seconds to every call.
    app.state.encoder = EmbeddingEncoder(os.getenv("MODEL_NAME", DEFAULT_MODEL))
    yield


app = FastAPI(title="Resume-Job Match API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_resume(
    request: Request,
    resume: UploadFile = File(..., description="Resume as a PDF file"),
    job_description: str = Form(...),
) -> AnalyzeResponse:
    if not (resume.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="The resume must be a PDF file.")

    data = resume.file.read(MAX_RESUME_BYTES + 1)
    if len(data) > MAX_RESUME_BYTES:
        raise HTTPException(status_code=413, detail="The resume is larger than 5 MB.")

    if len(job_description.strip()) < MIN_JOB_CHARS:
        raise HTTPException(status_code=422, detail="The job description is too short.")

    try:
        resume_text = extract_pdf_text(data)
    except Exception as exc:  # pypdf raises several exception types for bad files
        raise HTTPException(status_code=422, detail="Could not read that PDF.") from exc

    if not resume_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No text found in the PDF. Scanned resumes need OCR before analysis.",
        )

    report = analyze(resume_text, job_description, request.app.state.encoder.encode)
    return AnalyzeResponse(
        fit_score=round(report.fit_score, 4),
        matched_skills=report.matched_skills,
        missing_skills=report.missing_skills,
        matches=[RequirementMatchOut(**asdict(m)) for m in report.matches],
    )