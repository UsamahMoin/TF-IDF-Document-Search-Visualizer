let corpus = null;
let stopWords = new Set();
let documentsByName = new Map();
let expandedResult = null;

const queryForm = document.querySelector("#query-form");
const queryInput = document.querySelector("#query-input");
const queryChips = [...document.querySelectorAll("[data-query]")];
const rawTokensElement = document.querySelector("#raw-tokens");
const filteredTokensElement = document.querySelector("#filtered-tokens");
const stemmedTokensElement = document.querySelector("#stemmed-tokens");
const queryWeightsElement = document.querySelector("#query-weights");
const resultList = document.querySelector("#result-list");
const rankingNote = document.querySelector("#ranking-note");
const bestScore = document.querySelector("#best-score");
const timelineChart = document.querySelector("#timeline-chart");
const termInput = document.querySelector("#term-input");
const inspectTermButton = document.querySelector("#inspect-term");
const termChart = document.querySelector("#term-chart");

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatNumber(value) {
  return new Intl.NumberFormat("en-US").format(value);
}

function tokenize(text) {
  return text.toLowerCase().match(/[a-zA-Z]+/g) ?? [];
}

function stemToken(token) {
  return corpus.stemLookup[token] ?? token;
}

function processQuery(query) {
  const rawTokens = tokenize(query);
  const filteredTokens = rawTokens.filter((token) => !stopWords.has(token));
  const stems = filteredTokens.map(stemToken);
  const counts = new Map();

  stems.forEach((stem) => counts.set(stem, (counts.get(stem) ?? 0) + 1));

  const localWeights = new Map(
    [...counts.entries()].map(([stem, count]) => [stem, 1 + Math.log10(count)]),
  );
  const norm = Math.sqrt(
    [...localWeights.values()].reduce((sum, weight) => sum + weight ** 2, 0),
  );
  const weights = new Map(
    [...localWeights.entries()].map(([stem, weight]) => [
      stem,
      norm === 0 ? 0 : weight / norm,
    ]),
  );

  return { rawTokens, filteredTokens, stems, weights };
}

function renderTokens(container, tokens, stopWordMode = false) {
  if (tokens.length === 0) {
    container.innerHTML = '<span class="token">none</span>';
    return;
  }

  container.innerHTML = tokens
    .map((token) => {
      const stopClass = stopWordMode && stopWords.has(token) ? " is-stop" : "";
      return `<span class="token${stopClass}">${escapeHtml(token)}</span>`;
    })
    .join("");
}

function renderQueryWeights(weights) {
  if (weights.size === 0) {
    queryWeightsElement.innerHTML = '<span class="token">empty vector</span>';
    return;
  }

  const maxWeight = Math.max(...weights.values());
  queryWeightsElement.innerHTML = [...weights.entries()]
    .map(([stem, weight]) => `
      <div class="query-weight">
        <span>${escapeHtml(stem)}</span>
        <div class="weight-track"><div class="weight-fill" style="width:${(weight / maxWeight) * 100}%"></div></div>
        <span>${weight.toFixed(4)}</span>
      </div>
    `)
    .join("");
}

function rankDocuments(weights) {
  const scores = new Map(corpus.documents.map((document) => [document.filename, 0]));
  const contributions = new Map(corpus.documents.map((document) => [document.filename, []]));

  weights.forEach((queryWeight, stem) => {
    const term = corpus.terms[stem];
    if (!term) return;

    term.postings.forEach(({ doc, weight: documentWeight }) => {
      const contribution = queryWeight * documentWeight;
      scores.set(doc, scores.get(doc) + contribution);
      contributions.get(doc).push({
        stem,
        queryWeight,
        documentWeight,
        contribution,
      });
    });
  });

  return corpus.documents
    .map((document) => ({
      document,
      score: scores.get(document.filename),
      contributions: contributions
        .get(document.filename)
        .sort((first, second) => second.contribution - first.contribution),
    }))
    .sort((first, second) => second.score - first.score || first.document.year - second.document.year);
}

function renderRanking(ranking, processed) {
  const topScore = ranking[0]?.score ?? 0;
  const matchedTerms = [...processed.weights.keys()].filter((stem) => corpus.terms[stem]);
  const unmatchedTerms = [...processed.weights.keys()].filter((stem) => !corpus.terms[stem]);

  bestScore.textContent = topScore.toFixed(4);

  if (processed.weights.size === 0) {
    rankingNote.textContent = "The query contains no indexable terms after stop-word removal.";
  } else if (matchedTerms.length === 0) {
    rankingNote.textContent = "None of the query stems occur in this 15-document corpus.";
  } else {
    const matchText = `${matchedTerms.length} of ${processed.weights.size} query ${processed.weights.size === 1 ? "term" : "terms"} matched the corpus`;
    rankingNote.textContent = unmatchedTerms.length
      ? `${matchText}; unmatched: ${unmatchedTerms.join(", ")}.`
      : `${matchText}. Select a result to inspect score contributions.`;
  }

  resultList.innerHTML = ranking
    .map(({ document, score, contributions }, index) => {
      const width = topScore > 0 ? (score / topScore) * 100 : 0;
      const matchTerms = contributions
        .slice(0, 3)
        .map(({ stem }) => `<span class="match-term">${escapeHtml(stem)}</span>`)
        .join("");
      const contributionDetails = contributions.length
        ? contributions
            .map(({ stem, queryWeight, documentWeight, contribution }) => `
              <span>${escapeHtml(stem)}: ${queryWeight.toFixed(4)} × ${documentWeight.toFixed(4)} = ${contribution.toFixed(4)}</span>
            `)
            .join("")
        : "<span>No shared query terms</span>";

      return `
        <article
          class="result-row${expandedResult === document.filename ? " is-expanded" : ""}"
          data-result="${escapeHtml(document.filename)}"
          role="button"
          tabindex="0"
          aria-expanded="${expandedResult === document.filename}"
        >
          <span class="result-rank">${index + 1}</span>
          <div class="result-name">
            <strong>${escapeHtml(document.president)}</strong>
            <span>${document.year} · ${escapeHtml(document.filename)}</span>
          </div>
          <div class="score-area">
            <div class="match-terms">${matchTerms}</div>
            <div class="score-track"><div class="score-fill" style="width:${width}%"></div></div>
          </div>
          <span class="result-score">${score.toFixed(6)}</span>
          <div class="result-detail">
            <p>${escapeHtml(capitalize(document.excerpt))}</p>
            <div class="contribution-list">${contributionDetails}</div>
          </div>
        </article>
      `;
    })
    .join("");

  bindResultInteractions(ranking, processed);
}

function bindResultInteractions(ranking, processed) {
  document.querySelectorAll("[data-result]").forEach((row) => {
    const toggle = () => {
      expandedResult = expandedResult === row.dataset.result ? null : row.dataset.result;
      renderRanking(ranking, processed);
    };

    row.addEventListener("click", toggle);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        toggle();
      }
    });
  });
}

function runQuery(query) {
  if (!corpus) return;

  const processed = processQuery(query);
  renderTokens(rawTokensElement, processed.rawTokens, true);
  renderTokens(filteredTokensElement, processed.filteredTokens);
  renderTokens(stemmedTokensElement, processed.stems);
  renderQueryWeights(processed.weights);
  expandedResult = null;
  renderRanking(rankDocuments(processed.weights), processed);

  queryChips.forEach((chip) => {
    chip.classList.toggle(
      "is-active",
      chip.dataset.query.toLowerCase() === query.trim().toLowerCase(),
    );
  });
}

function capitalize(text) {
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : "";
}

function renderMetrics() {
  document.querySelector("#metric-documents").textContent = formatNumber(corpus.documentCount);
  document.querySelector("#metric-words").textContent = formatNumber(corpus.totalWordCount);
  document.querySelector("#metric-indexed").textContent = formatNumber(corpus.totalIndexedTokenCount);
  document.querySelector("#metric-vocabulary").textContent = formatNumber(corpus.vocabularySize);
}

function renderTimeline() {
  const maxWords = Math.max(...corpus.documents.map((document) => document.wordCount));
  timelineChart.innerHTML = corpus.documents
    .map((document, index) => {
      const height = Math.max(10, (document.wordCount / maxWords) * 174);
      return `
        <div class="timeline-item">
          <button
            class="timeline-bar${index === 0 ? " is-active" : ""}"
            type="button"
            data-document="${escapeHtml(document.filename)}"
            data-label="${formatNumber(document.wordCount)} words"
            style="height:${height}px"
            aria-label="${escapeHtml(document.president)}, ${document.year}: ${formatNumber(document.wordCount)} words"
          ></button>
          <span class="timeline-year">${document.year}</span>
        </div>
      `;
    })
    .join("");

  document.querySelectorAll("[data-document]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-document]").forEach((candidate) => {
        candidate.classList.toggle("is-active", candidate === button);
      });
      renderDocumentInspector(documentsByName.get(button.dataset.document));
    });
  });

  renderDocumentInspector(corpus.documents[0]);
}

function renderDocumentInspector(selectedDocument) {
  documentYear.textContent = selectedDocument.year;
  documentPresident.textContent = selectedDocument.president;
  documentFilename.textContent = selectedDocument.filename;
  documentWords.textContent = formatNumber(selectedDocument.wordCount);
  documentIndexed.textContent = formatNumber(selectedDocument.indexedTokenCount);
  documentUnique.textContent = formatNumber(selectedDocument.uniqueTermCount);
  documentExcerpt.textContent = capitalize(selectedDocument.excerpt);
  documentTopTerms.innerHTML = selectedDocument.topTerms
    .map(({ term, weight }) => `<span class="top-term">${escapeHtml(term)} · ${weight.toFixed(4)}</span>`)
    .join("");
}

const documentYear = document.querySelector("#document-year");
const documentPresident = document.querySelector("#document-president");
const documentFilename = document.querySelector("#document-filename");
const documentWords = document.querySelector("#document-words");
const documentIndexed = document.querySelector("#document-indexed");
const documentUnique = document.querySelector("#document-unique");
const documentExcerpt = document.querySelector("#document-excerpt");
const documentTopTerms = document.querySelector("#document-top-terms");

function inspectTerm(rawValue) {
  if (!corpus) return;

  const rawTerm = tokenize(rawValue)[0] ?? "";
  const stem = rawTerm ? stemToken(rawTerm) : "";
  const term = corpus.terms[stem];

  document.querySelector("#term-stem").textContent = rawTerm
    ? `${rawTerm} → ${stem}`
    : "enter a term";

  if (!term) {
    document.querySelector("#term-df").textContent = "0 / 15";
    document.querySelector("#term-cf").textContent = "0";
    document.querySelector("#term-idf").textContent = "not indexed";
    document.querySelector("#idf-formula").textContent = "Term does not occur in this corpus";
    document.querySelector("#term-chart-caption").textContent = "No matching documents";
    termChart.innerHTML = '<div class="ranking-note">Try “union”, “war”, “public”, or “military”.</div>';
    return;
  }

  document.querySelector("#term-df").textContent = `${term.df} / ${corpus.documentCount}`;
  document.querySelector("#term-cf").textContent = formatNumber(term.cf);
  document.querySelector("#term-idf").textContent = term.idf.toFixed(6);
  document.querySelector("#idf-formula").textContent =
    `log10(${corpus.documentCount} / ${term.df}) = ${term.idf.toFixed(6)}`;
  document.querySelector("#term-chart-caption").textContent =
    `${term.df} ${term.df === 1 ? "document contains" : "documents contain"} “${stem}”`;

  const maxWeight = Math.max(...term.postings.map((posting) => posting.weight));
  termChart.innerHTML = term.postings
    .map((posting) => {
      const document = documentsByName.get(posting.doc);
      return `
        <div class="term-bar-row">
          <div class="term-bar-label">
            <strong>${escapeHtml(document.president)}</strong>
            <span>${document.year}</span>
          </div>
          <div class="term-weight-track">
            <div class="term-weight-fill" style="width:${(posting.weight / maxWeight) * 100}%"></div>
          </div>
          <span class="term-bar-value">${posting.weight.toFixed(4)}</span>
        </div>
      `;
    })
    .join("");
}

async function initialize() {
  try {
    const response = await fetch("corpus-index.json");
    if (!response.ok) throw new Error(`Index request failed with ${response.status}`);

    corpus = await response.json();
    stopWords = new Set(corpus.stopWords);
    documentsByName = new Map(
      corpus.documents.map((document) => [document.filename, document]),
    );

    renderMetrics();
    renderTimeline();
    runQuery(queryInput.value);
    inspectTerm(termInput.value);
  } catch (error) {
    rankingNote.textContent = "The corpus index could not be loaded.";
    resultList.innerHTML = `<div class="ranking-note">${escapeHtml(error.message)}</div>`;
    console.error(error);
  }
}

queryForm.addEventListener("submit", (event) => {
  event.preventDefault();
  runQuery(queryInput.value);
});

queryChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    queryInput.value = chip.dataset.query;
    runQuery(chip.dataset.query);
  });
});

inspectTermButton.addEventListener("click", () => inspectTerm(termInput.value));
termInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    inspectTerm(termInput.value);
  }
});

initialize();
