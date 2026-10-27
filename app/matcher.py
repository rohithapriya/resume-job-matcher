"""Fit scoring.

For each job requirement, find the most similar resume line (cosine similarity).
The fit score is the average of those best similarities, clipped to [0, 1].

Scoring per line instead of embedding the whole resume keeps every input short
enough for the encoder's token limit, and it gives the per-line matches the UI shows.

The `encode` argument is any callable that maps a list of strings to an array
of L2-normalised row vectors. EmbeddingEncoder provides one. The TF-IDF
baseline in eval/evaluate.py provides another, so both go through identical logic.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from app.skills import find_skills
from app.text_utils import split_lines, split_requirements

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

Encoder = Callable[[list[str]], np.ndarray]


class EmbeddingEncoder:
    """Sentence-transformers encoder. Load it once and reuse it."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer  # heavy import, so keep it local

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


@dataclass
class RequirementMatch:
    requirement: str
    best_resume_line: str
    similarity: float


@dataclass
class MatchReport:
    fit_score: float
    matches: list[RequirementMatch] = field(default_factory=list)
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)


def score_pair(
    resume_text: str, job_text: str, encode: Encoder
) -> tuple[float, list[RequirementMatch]]:
    """Return (fit_score, per-requirement matches) for one resume and one job."""
    resume_lines = split_lines(resume_text)
    requirements = split_requirements(job_text)
    if not resume_lines or not requirements:
        return 0.0, []

    # One encoder call for everything keeps batching efficient
    vectors = np.asarray(encode(requirements + resume_lines))
    n_req = len(requirements)
    req_vecs, line_vecs = vectors[:n_req], vectors[n_req:]

    similarity = req_vecs @ line_vecs.T  # shape (requirements, resume lines)
    best_idx = similarity.argmax(axis=1)
    best_sim = np.clip(similarity.max(axis=1), 0.0, 1.0)

    matches = [
        RequirementMatch(requirement=req, best_resume_line=resume_lines[j], similarity=float(s))
        for req, j, s in zip(requirements, best_idx, best_sim)
    ]
    return float(best_sim.mean()), matches


def analyze(resume_text: str, job_text: str, encode: Encoder) -> MatchReport:
    """Full report: fit score, closest resume line per requirement, and skill gaps."""
    fit, matches = score_pair(resume_text, job_text, encode)

    job_skills = find_skills(job_text)
    resume_skills = set(find_skills(resume_text))
    return MatchReport(
        fit_score=fit,
        matches=matches,
        matched_skills=[s for s in job_skills if s in resume_skills],
        missing_skills=[s for s in job_skills if s not in resume_skills],
    )