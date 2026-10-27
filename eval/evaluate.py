"""Compare fit scores with human relevance labels.

Run from the project root:
    python -m eval.evaluate
    python -m eval.evaluate --model sentence-transformers/all-mpnet-base-v2

Expected layout:
    eval/labels.csv              columns: resume_file, job_file, label (integer 1-5)
    eval/data/resumes/*.pdf|txt  resume files named in labels.csv
    eval/data/jobs/*.txt         job descriptions named in labels.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer

from app.matcher import DEFAULT_MODEL, EmbeddingEncoder, score_pair
from app.text_utils import extract_pdf_text, split_lines, split_requirements

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "data"


def load_resume(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_pdf_text(path.read_bytes())
    return path.read_text(encoding="utf-8")


def load_pairs(labels_path: Path) -> list[tuple[str, str, float]]:
    pairs = []
    with labels_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            resume = load_resume(DATA_DIR / "resumes" / row["resume_file"])
            job = (DATA_DIR / "jobs" / row["job_file"]).read_text(encoding="utf-8")
            pairs.append((resume, job, float(row["label"])))
    return pairs


def make_tfidf_encoder(pairs):
    """Fit TF-IDF on every resume line and requirement in the eval set.

    The labels are not used, so there is no leakage. Fitting IDF on the whole
    corpus gives the baseline a fair vocabulary instead of a per-pair one.
    """
    corpus: list[str] = []
    for resume, job, _ in pairs:
        corpus += split_lines(resume) + split_requirements(job)
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    vectorizer.fit(corpus)
    return lambda texts: vectorizer.transform(texts).toarray()


def bootstrap_difference(a, b, labels, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI for spearman(a, labels) - spearman(b, labels), resampling pairs."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        rho_a = spearmanr(a[idx], labels[idx])[0]
        rho_b = spearmanr(b[idx], labels[idx])[0]
        if np.isfinite(rho_a) and np.isfinite(rho_b):
            diffs.append(rho_a - rho_b)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=EVAL_DIR / "labels.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model name")
    args = parser.parse_args()

    pairs = load_pairs(args.labels)
    labels = np.array([label for _, _, label in pairs])
    print(f"{len(pairs)} labelled pairs. Label counts: {sorted(Counter(labels.tolist()).items())}")

    tfidf_encode = make_tfidf_encoder(pairs)
    tfidf_scores = np.array([score_pair(r, j, tfidf_encode)[0] for r, j, _ in pairs])

    embedder = EmbeddingEncoder(args.model)
    emb_scores = np.array([score_pair(r, j, embedder.encode)[0] for r, j, _ in pairs])

    rho_tfidf, p_tfidf = spearmanr(tfidf_scores, labels)
    rho_emb, p_emb = spearmanr(emb_scores, labels)
    low, high = bootstrap_difference(emb_scores, tfidf_scores, labels)

    print()
    print(f"TF-IDF baseline        Spearman rho = {rho_tfidf:.3f}  (p = {p_tfidf:.2g})")
    print(f"{args.model:<22} Spearman rho = {rho_emb:.3f}  (p = {p_emb:.2g})")
    print(
        f"Difference (embedding - TF-IDF) = {rho_emb - rho_tfidf:+.3f}, "
        f"95% bootstrap CI [{low:+.3f}, {high:+.3f}]"
    )
    if low <= 0 <= high:
        print("The CI includes zero, so this sample does not show that the embedding model is better.")


if __name__ == "__main__":
    main()