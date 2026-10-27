from app.skills import find_skills
from app.text_utils import split_lines, split_requirements


def test_finds_skills_and_aliases():
    text = "Built APIs with FastAPI and Postgres on AWS. Wrote JavaScript, not Java."
    skills = find_skills(text)
    for expected in ["fastapi", "postgresql", "aws", "javascript", "java"]:
        assert expected in skills


def test_java_does_not_match_inside_javascript():
    assert "java" not in find_skills("Experienced in JavaScript only")


def test_sql_does_not_match_inside_postgresql():
    assert "sql" not in find_skills("PostgreSQL administration")


def test_split_lines_rejoins_wrapped_bullets_and_drops_short_lines():
    text = "• Built a resume matching tool that\nscores fit with sentence embeddings\nShort\n"
    assert split_lines(text) == [
        "Built a resume matching tool that scores fit with sentence embeddings"
    ]


def test_split_requirements_breaks_sentences():
    job = "Build backend services in Python. Experience with Docker and Kubernetes is a plus."
    assert len(split_requirements(job)) == 2
