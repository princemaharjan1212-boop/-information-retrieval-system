"""
CISI Information Retrieval System
Model: TF-IDF + cosine similarity
Dataset files :
  - CISI.ALL (documents)
  - CISI.QRY (queries)
  - CISI.REL (relevance judgments)
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent


def parse_tagged_file(path, text_tags):
    """Parse CISI files whose records begin with .I and fields with tags such as .T/.W."""
    records = []
    current = None
    active_tag = None

    with open(path, "r", encoding="latin-1") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")
            match = re.match(r"^\.(\w+)\s*(.*)$", line)

            if match:
                tag, remainder = match.group(1), match.group(2)
                if tag == "I":
                    if current is not None:
                        records.append(current)
                    current = {"id": remainder.strip()}
                    active_tag = None
                elif current is not None and tag in text_tags:
                    current.setdefault(tag, [])
                    if remainder.strip():
                        current[tag].append(remainder.strip())
                    active_tag = tag
                else:
                    active_tag = None
            elif current is not None and active_tag is not None and line.strip():
                current.setdefault(active_tag, []).append(line.strip())

        if current is not None:
            records.append(current)

    for record in records:
        for key, value in list(record.items()):
            if key != "id" and isinstance(value, list):
                record[key] = " ".join(value)

    return records


def load_dataset():
    required = ["CISI.ALL", "CISI.QRY", "CISI.REL"]
    missing = [name for name in required if not (DATA_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset file(s): " + ", ".join(missing) +
            "\nPlace CISI.ALL, CISI.QRY, and CISI.REL in the same folder as this script."
        )

    raw_docs = parse_tagged_file(DATA_DIR / "CISI.ALL", {"T", "A", "W"})
    raw_queries = parse_tagged_file(DATA_DIR / "CISI.QRY", {"W"})

    docs = pd.DataFrame([
        {
            "doc_id": int(item["id"]),
            "title": item.get("T", ""),
            "abstract": item.get("W", ""),
            "text": (item.get("T", "") + " " + item.get("W", "")).strip()
        }
        for item in raw_docs if item.get("id", "").isdigit()
    ])

    query_df = pd.DataFrame([
        {"query_id": int(item["id"]), "query": item.get("W", "")}
        for item in raw_queries if item.get("id", "").isdigit()
    ])

    docs = docs.dropna(subset=["text"]).copy()
    docs["text"] = docs["text"].astype(str).str.strip()
    docs = docs[docs["text"].ne("")]
    docs = docs.drop_duplicates(subset="doc_id").reset_index(drop=True)
    return docs, query_df



def load_relevance():
    """Load CISI query-document relevance pairs.

    In CISI.REL, the first two columns are the query ID
    and relevant document ID. Other columns are not used
    as binary relevance labels.
    """
    path = DATA_DIR / "CISI.REL"
    rows = []

    with open(path, "r", encoding="latin-1") as file:
        for line in file:
            parts = line.split()

            if len(parts) < 2:
                continue

            # Standard CISI.REL entries contain groups of:
            # query_id, doc_id, 0, 0.000000.
            # Handle multiple groups on a line as well.
            is_four_column_format = (
                len(parts) >= 4
                and len(parts) % 4 == 0
            )

            if is_four_column_format:
                try:
                    groups = [
                        parts[i:i + 4]
                        for i in range(0, len(parts), 4)
                    ]

                    valid_groups = all(
                        len(group) == 4
                        and group[0].isdigit()
                        and group[1].isdigit()
                        and group[2] == "0"
                        and float(group[3]) >= 0
                        for group in groups
                    )
                except ValueError:
                    valid_groups = False

                if valid_groups:
                    for group in groups:
                        rows.append({
                            "query_id": int(group[0]),
                            "doc_id": int(group[1]),
                            "relevance": 1.0
                        })
                    continue

            # Also support files with two-column entries.
            try:
                rows.append({
                    "query_id": int(parts[0]),
                    "doc_id": int(parts[1]),
                    "relevance": 1.0
                })
            except ValueError:
                continue

    if not rows:
        raise ValueError(
            "No relevance judgments could be parsed from CISI.REL."
        )

    relevance = pd.DataFrame(rows)
    relevance = relevance.drop_duplicates(
        subset=["query_id", "doc_id"]
    ).reset_index(drop=True)

    print(f"Relevance judgments loaded: {len(relevance)}")
    print(
        "Queries with relevance judgments:",
        relevance["query_id"].nunique()
    )

    return relevance


def precision_recall_at_k(ranked_ids, relevant_ids, k=10):
    top_k_ids = ranked_ids[:k]
    hits = sum(doc_id in relevant_ids for doc_id in top_k_ids)
    precision = hits / k if k else 0.0
    recall = hits / len(relevant_ids) if relevant_ids else 0.0
    return precision, recall


def average_precision(ranked_ids, relevant_ids, k=100):
    if not relevant_ids:
        return np.nan
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant_ids:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant_ids), k)


def main():
    print("Loading CISI files...")
    documents, queries = load_dataset()
    relevance = load_relevance()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(documents["text"])

    print(f"Documents indexed: {len(documents)}")
    print(f"Queries loaded: {len(queries)}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    doc_ids = documents["doc_id"].to_numpy()

    def search(query_text, top_k=10):
        query_vector = vectorizer.transform([str(query_text)])
        scores = cosine_similarity(query_vector, tfidf_matrix).ravel()
        ranked_indices = np.argsort(-scores, kind="stable")
        ranked_indices = [idx for idx in ranked_indices if scores[idx] > 0][:top_k]
        if not ranked_indices:
            return pd.DataFrame(columns=["rank", "doc_id", "score", "title", "snippet"])
        results = documents.iloc[ranked_indices][["doc_id", "title", "text"]].copy()
        results.insert(0, "rank", range(1, len(results) + 1))
        results["score"] = [float(scores[idx]) for idx in ranked_indices]
        results["snippet"] = results["text"].str.slice(0, 240)
        return results[["rank", "doc_id", "score", "title", "snippet"]].reset_index(drop=True)

    print("\nExample search")
    example_query = "information retrieval and library science"
    example_results = search(example_query, top_k=10)
    print("Query:", example_query)
    if example_results.empty:
        print("No matching documents found.")
    else:
        print(example_results.to_string(index=False))
    example_results.to_csv(DATA_DIR / "cisi_example_search_results.csv", index=False)
    print("\nSaved cisi_example_search_results.csv")

    # Evaluate each query that has at least one known relevant document.
    precision_values, recall_values, ap_values = [], [], []
    for query_row in queries.itertuples(index=False):
        judged = relevance[
            (relevance["query_id"] == query_row.query_id) &
            (relevance["relevance"] > 0)
        ]
        relevant_ids = set(judged["doc_id"].astype(int))
        if not relevant_ids:
            continue

        result = search(query_row.query, top_k=100)
        ranked_ids = result["doc_id"].astype(int).tolist()
        p_at_10, r_at_10 = precision_recall_at_k(ranked_ids, relevant_ids, k=10)
        ap_at_100 = average_precision(ranked_ids, relevant_ids, k=100)
        precision_values.append(p_at_10)
        recall_values.append(r_at_10)
        ap_values.append(ap_at_100)

    metrics = {
        "Precision@10": float(np.mean(precision_values)) if precision_values else float("nan"),
        "Recall@10": float(np.mean(recall_values)) if recall_values else float("nan"),
        "MAP@100": float(np.nanmean(ap_values)) if ap_values else float("nan"),
        "Evaluated queries": len(precision_values)
    }
    print("\nEvaluation metrics")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")

    pd.DataFrame([metrics]).to_csv(DATA_DIR / "cisi_evaluation_metrics.csv", index=False)
    print("\nSaved cisi_evaluation_metrics.csv")

    # Optional interactive searching from terminal.
    while True:
        try:
            user_query = input("\nEnter a search query (or type 'exit' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if user_query.lower() in {"exit", "quit"}:
            break
        if not user_query:
            print("Please enter a non-empty query.")
            continue
        results = search(user_query, top_k=10)
        if results.empty:
            print("No matching documents found.")
        else:
            print(results.to_string(index=False))


if __name__ == "__main__":
    main()
