"""Curated skill vocabulary and whole-word skill detection.

Keys are the names shown in the UI. Values are the spellings that count as a
mention. Add entries to cover more skills. Short ambiguous words such as "go"
and "r" are left out on purpose, because they produce false matches.
"""
from __future__ import annotations

import re

SKILL_ALIASES: dict[str, list[str]] = {
    # Languages
    "python": ["python"],
    "java": ["java"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript"],
    "c++": ["c++"],
    "c#": ["c#"],
    "rust": ["rust"],
    "scala": ["scala"],
    "sql": ["sql"],
    # Frameworks and libraries
    "react": ["react", "reactjs"],
    "node.js": ["node.js", "nodejs"],
    "fastapi": ["fastapi"],
    "flask": ["flask"],
    "django": ["django"],
    "spring boot": ["spring boot"],
    "streamlit": ["streamlit"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "pytorch": ["pytorch"],
    "tensorflow": ["tensorflow"],
    "pytest": ["pytest"],
    # Data and databases
    "postgresql": ["postgresql", "postgres"],
    "mysql": ["mysql"],
    "mongodb": ["mongodb"],
    "redis": ["redis"],
    "spark": ["spark", "pyspark", "apache spark"],
    "kafka": ["kafka"],
    "airflow": ["airflow", "apache airflow"],
    "tableau": ["tableau"],
    "power bi": ["power bi", "powerbi"],
    "excel": ["excel"],
    # Cloud and DevOps
    "aws": ["aws", "amazon web services"],
    "gcp": ["gcp", "google cloud"],
    "azure": ["azure"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "terraform": ["terraform"],
    "ci/cd": ["ci/cd", "cicd"],
    "linux": ["linux"],
    "git": ["git", "github", "gitlab"],
    # ML and AI
    "machine learning": ["machine learning"],
    "deep learning": ["deep learning"],
    "nlp": ["nlp", "natural language processing"],
    "llm": ["llm", "llms", "large language model", "large language models"],
    "rag": ["rag", "retrieval-augmented generation", "retrieval augmented generation"],
    "embeddings": ["embeddings", "sentence embeddings"],
    "statistics": ["statistics", "statistical analysis"],
    "a/b testing": ["a/b testing", "ab testing"],
    "data visualization": ["data visualization", "data visualisation"],
    # APIs and architecture
    "rest apis": ["rest api", "rest apis", "restful"],
    "graphql": ["graphql"],
    "microservices": ["microservices", "microservice"],
    "system design": ["system design"],
    # Process and tooling
    "agile": ["agile", "scrum"],
    "jira": ["jira"],
    "unit testing": ["unit testing", "unit tests"],
    "figma": ["figma"],
}


def _build_patterns() -> list[tuple[str, re.Pattern[str]]]:
    patterns = []
    for name, aliases in SKILL_ALIASES.items():
        # Longest alias first so multi-word aliases win over their parts
        alternation = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))
        # Whole-token match: "java" must not match "JavaScript", and "sql" must not match "PostgreSQL"
        patterns.append(
            (name, re.compile(rf"(?<![\w+#.])(?:{alternation})(?![\w+#])", re.IGNORECASE))
        )
    return patterns


_PATTERNS = _build_patterns()


def find_skills(text: str) -> list[str]:
    """Return the canonical names of the skills mentioned in text, sorted."""
    return sorted(name for name, pattern in _PATTERNS if pattern.search(text))