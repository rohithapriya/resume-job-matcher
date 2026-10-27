# Resume–Job Match Analyzer

Upload a PDF resume and paste a job description. The app returns a fit score from 0 to 1, the skills the job asks for that the resume doesn't mention, and the resume line closest to each job requirement.

## How it works

- Extracts text from the PDF resume with pypdf.
- Splits the job description into requirement sentences and the resume into lines.
- Embeds both with the sentence-transformers model `all-MiniLM-L6-v2` and compares them using cosine similarity.
- The fit score is the average of each requirement's best match.
- Skill gaps come from a curated skill list in `app/skills.py`.

## Tech stack

Python, FastAPI, Streamlit, sentence-transformers, PyTorch, pypdf, scikit-learn, SciPy, pytest, Docker

## Project structure

```
app/          scoring logic, skill detection, FastAPI endpoint
ui/           Streamlit front end
eval/         evaluation script and labels (in progress)
tests/        unit tests
Dockerfile    container setup (not yet tested)
```

## Run locally

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.api:app --reload --port 8000
streamlit run ui/streamlit_app.py
```

Open http://localhost:8501 in your browser.

## Run tests

```bash
python -m pytest
```

## Status

- **Working:** PDF parsing, embedding-based scoring, skill gap detection, FastAPI endpoint, Streamlit interface, unit tests.
- **In progress:** evaluation of fit scores against hand-labelled resume–job pairs, compared with a TF-IDF baseline.
- **Not yet tested:** Docker image build.
