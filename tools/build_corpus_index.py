#!/usr/bin/env python3
"""Build the static search index consumed by the GitHub Pages visualizer."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from nltk.stem.porter import PorterStemmer


ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = ROOT / "US_Inaugural_Addresses"
OUTPUT_PATH = ROOT / "corpus-index.json"

# NLTK's English stopword corpus, embedded so the build does not require a
# separate nltk.download("stopwords") step.
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "she's", "her",
    "hers", "herself", "it", "it's", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "that'll", "these", "those", "am", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because",
    "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "just", "don", "don't", "should", "should've", "now", "d", "ll", "m",
    "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't",
    "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn",
    "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn",
    "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't",
    "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won",
    "won't", "wouldn", "wouldn't",
}

PRESIDENT_NAMES = {
    "washington": "George Washington",
    "adams_john": "John Adams",
    "jefferson": "Thomas Jefferson",
    "madison": "James Madison",
    "monroe": "James Monroe",
    "adams_john_quincy": "John Quincy Adams",
    "jackson": "Andrew Jackson",
    "van_buren": "Martin Van Buren",
    "harrison": "William Henry Harrison",
    "polk": "James K. Polk",
}

TOKEN_PATTERN = re.compile(r"[a-zA-Z]+")
stemmer = PorterStemmer()


def parse_filename(filename: str) -> tuple[str, int]:
    parts = Path(filename).stem.split("_")
    president_key = "_".join(parts[1:-1])
    year = parts[-1]
    return PRESIDENT_NAMES[president_key], int(year)


def clean_excerpt(text: str, length: int = 250) -> str:
    compact = " ".join(text.split())
    return compact[:length].rstrip() + ("..." if len(compact) > length else "")


def build_index() -> dict:
    raw_documents: dict[str, str] = {}
    tokenized_documents: dict[str, list[str]] = {}
    raw_token_counts: dict[str, int] = {}
    stem_lookup: dict[str, str] = {}

    for path in sorted(CORPUS_ROOT.glob("*.txt")):
        content = path.read_text(encoding="windows-1252").lower()
        raw_tokens = TOKEN_PATTERN.findall(content)
        filtered_words = [token for token in raw_tokens if token not in STOP_WORDS]
        stems = [stemmer.stem(token) for token in filtered_words]

        raw_documents[path.name] = content
        raw_token_counts[path.name] = len(raw_tokens)
        tokenized_documents[path.name] = stems
        stem_lookup.update({token: stemmer.stem(token) for token in filtered_words})

    document_frequency: Counter[str] = Counter()
    collection_frequency: Counter[str] = Counter()
    for tokens in tokenized_documents.values():
        document_frequency.update(set(tokens))
        collection_frequency.update(tokens)

    document_count = len(tokenized_documents)
    idf = {
        term: math.log10(document_count / frequency)
        for term, frequency in document_frequency.items()
    }

    documents = []
    postings: dict[str, list[dict]] = defaultdict(list)

    for filename, tokens in tokenized_documents.items():
        counts = Counter(tokens)
        unnormalized = {
            term: (1 + math.log10(count)) * idf[term]
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(weight * weight for weight in unnormalized.values()))
        vector = {
            term: weight / norm
            for term, weight in unnormalized.items()
        }
        president, year = parse_filename(filename)
        top_terms = sorted(vector.items(), key=lambda item: item[1], reverse=True)[:8]

        for term, weight in vector.items():
            postings[term].append({"doc": filename, "weight": round(weight, 12)})

        documents.append(
            {
                "filename": filename,
                "president": president,
                "year": year,
                "wordCount": raw_token_counts[filename],
                "indexedTokenCount": len(tokens),
                "uniqueTermCount": len(counts),
                "excerpt": clean_excerpt(raw_documents[filename]),
                "topTerms": [
                    {"term": term, "weight": round(weight, 12)}
                    for term, weight in top_terms
                ],
            }
        )

    terms = {
        term: {
            "df": document_frequency[term],
            "cf": collection_frequency[term],
            "idf": round(idf[term], 12),
            "postings": sorted(
                postings[term],
                key=lambda item: item["weight"],
                reverse=True,
            ),
        }
        for term in sorted(idf)
    }

    return {
        "scheme": "ltc.lnc",
        "documentCount": document_count,
        "totalWordCount": sum(raw_token_counts.values()),
        "totalIndexedTokenCount": sum(len(tokens) for tokens in tokenized_documents.values()),
        "vocabularySize": len(idf),
        "stopWordCount": len(STOP_WORDS),
        "documents": sorted(documents, key=lambda document: document["year"]),
        "terms": terms,
        "stemLookup": dict(sorted(stem_lookup.items())),
        "stopWords": sorted(STOP_WORDS),
    }


if __name__ == "__main__":
    OUTPUT_PATH.write_text(
        json.dumps(build_index(), separators=(",", ":"), ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
