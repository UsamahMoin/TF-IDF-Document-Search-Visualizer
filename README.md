# TF-IDF Document Search Visualizer

This project implements an `ltc.lnc` vector-space retrieval model over 15 early
U.S. presidential inaugural addresses. It includes the original Python and
Jupyter Notebook implementation plus an interactive GitHub Pages visualizer.

Live page: https://usamahmoin.github.io/TF-IDF-Document-Search-Visualizer/

## What the visualizer shows

- Live query tokenization, stop-word removal, and Porter stemming.
- Logarithmic query term-frequency weights with cosine normalization.
- Ranked cosine-similarity results over the actual 15-document corpus.
- Per-term score contributions for every ranked document.
- Corpus length, document statistics, excerpts, and highest-weighted terms.
- An IDF inspector showing document frequency and normalized document weights.
- A visual explanation of the SMART `ltc.lnc` weighting scheme.

## Retrieval model

Document vectors use:

```text
ltc = logarithmic TF × logarithmic IDF, cosine normalized
```

Query vectors use:

```text
lnc = logarithmic TF × no IDF, cosine normalized
```

The final ranking score is the dot product of the normalized document and query
vectors, which is their cosine similarity.

## Project files

- `index.html` - GitHub Pages entry point.
- `style.css` - shared blue visual theme and responsive layout.
- `script.js` - interactive retrieval, ranking, charts, and inspectors.
- `corpus-index.json` - generated TF-IDF index consumed by the browser.
- `tools/build_corpus_index.py` - deterministic static index generator.
- `UsamahMoinMohammed.py` - original command-line implementation.
- `P1 Intro.ipynb` - assignment notebook.
- `US_Inaugural_Addresses/` - the 15-document source corpus.

## Run the web page locally

```bash
python3 -m http.server 8000
```

Then open:

```text
http://127.0.0.1:8000/
```

## Rebuild the browser index

Install NLTK, then run:

```bash
pip install nltk
python3 tools/build_corpus_index.py
```

The stopword list is embedded in the generator, so rebuilding does not require
an additional NLTK corpus download.

## Run the original script

```bash
pip install nltk
python3 UsamahMoinMohammed.py
```

The script uses the installed NLTK stopword corpus when available and otherwise
falls back to the same embedded 179-word English list used by the index builder.

## Research basis

- Gerard Salton, Anita Wong, and Chung-Shu Yang, "A Vector Space Model for Automatic Indexing" (1975).
- Gerard Salton and Christopher Buckley, "Term-Weighting Approaches in Automatic Text Retrieval" (1988).
- Martin Porter, "An Algorithm for Suffix Stripping" (1980).

## GitHub Pages

Publish from the `main` branch at the repository root.
