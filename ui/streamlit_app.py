"""Streamlit front end. It sends the resume and job description to the FastAPI service."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Resume–Job Match", layout="wide")
st.title("Resume–Job Match Analyzer")
st.caption("Upload a PDF resume and paste a job description to see how well they line up.")

left, right = st.columns(2)
with left:
    resume_file = st.file_uploader("Resume (PDF)", type=["pdf"])
with right:
    job_text = st.text_area(
        "Job description", height=280, placeholder="Paste the full job posting here"
    )

ready = resume_file is not None and len(job_text.strip()) >= 30
if st.button("Analyze fit", type="primary", disabled=not ready):
    with st.spinner("Scoring..."):
        try:
            response = requests.post(
                f"{API_URL}/analyze",
                files={"resume": (resume_file.name, resume_file.getvalue(), "application/pdf")},
                data={"job_description": job_text},
                timeout=120,
            )
        except requests.RequestException as exc:
            st.error(f"Could not reach the API at {API_URL}: {exc}")
            st.stop()

    if response.status_code != 200:
        st.error(response.json().get("detail", response.text))
        st.stop()

    result = response.json()

    st.metric(
        "Fit score",
        f"{result['fit_score']:.2f}",
        help="Cosine-similarity based, from 0 to 1. Use it to compare jobs, not as a probability.",
    )

    col_missing, col_matched = st.columns(2)
    with col_missing:
        st.subheader("Asked for, not in the resume")
        if result["missing_skills"]:
            for skill in result["missing_skills"]:
                st.markdown(f"- {skill}")
        else:
            st.write("None found in the skill list.")
    with col_matched:
        st.subheader("Asked for and already in the resume")
        if result["matched_skills"]:
            for skill in result["matched_skills"]:
                st.markdown(f"- {skill}")
        else:
            st.write("No overlap found in the skill list.")

    st.subheader("Closest resume line for each job requirement")
    rows = sorted(result["matches"], key=lambda m: m["similarity"], reverse=True)
    st.dataframe(rows)

    with st.expander("How is this scored?"):
        st.markdown(
            "Each sentence in the job description is compared with every line of the resume "
            "using sentence embeddings. The fit score is the average of the best match for each "
            "requirement. Skills come from a curated list, so a skill missing from that list "
            "will not show up."
        )
