import time
from pathlib import Path
from flask import Flask, request, render_template_string
import search as search

INDEX_FOLDER = Path("dev_index")
INDEX_FILE_NAME = "index_merged.jsonl"
DOC_TABLE_NAME = "doc_table.tsv"
PAGERANK_FILE_NAME = "pagerank.json"
TOP_K_DEFAULT = 10

app = Flask(__name__)
index_path = INDEX_FOLDER / INDEX_FILE_NAME
doc_table_path = INDEX_FOLDER / DOC_TABLE_NAME
pagerank_path = INDEX_FOLDER / PAGERANK_FILE_NAME
lexicon_path = index_path.with_suffix(".lexicon.json")
print("Loading document table...")
DOCID_TO_URL = search.load_doc_table(doc_table_path)
NUM_DOCS = len(DOCID_TO_URL)
print(f"Loaded {NUM_DOCS} documents.")
print("Loading PageRank scores...")
PAGERANK_SCORES = search.load_pagerank(pagerank_path, NUM_DOCS)
print("Building or loading lexicon (term → file offset)...")
TERM_OFFSETS = search.load_lexicon(index_path, lexicon_path)
print(f"Lexicon loaded for {len(TERM_OFFSETS)} terms.")

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>ICS Search Engine</title>
    <style>
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            margin: 0;
            padding: 0;
            background: #FAF9F6;
            color: #e5e7eb;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem 1.5rem 3rem;
        }
        h1 {
            font-size: 1.8rem;
            margin-bottom: 1rem;
            color: #6890BD;
        }
        form {
            margin-bottom: 1.5rem;
        }
        .search-bar {
            display: flex;
            gap: 0.5rem;
        }
        input[type="text"] {
            flex: 1;
            padding: 0.6rem 0.8rem;
            border-radius: 999px;
            border: 1px solid #374151;
            background: #FAF9F6;
            color: #000000;
            outline: none;
        }
        input[type="text"]::placeholder {
            color: #6b7280;
        }
        button {
            padding: 0.6rem 1.1rem;
            border-radius: 999px;
            border: none;
            background: #fbbf24;
            color: #000000;
            font-weight: 600;
            cursor: pointer;
        }
        button:hover {
            background: #facc15;
        }
        .meta {
            font-size: 0.85rem;
            color: #000000;
            margin-top: 0.3rem;
        }
        .result-list {
            margin-top: 1.5rem;
        }
        .result {
            padding: 0.9rem 0.3rem;
            border-bottom: 1px solid #1f2937;
        }
        .result-title {
            font-size: 1rem;
            font-weight: 600;
            color: #60a5fa;
            text-decoration: none;
        }
        .result-title:hover {
            text-decoration: underline;
        }
        .result-url {
            font-size: 0.8rem;
            color: #000000;
            margin-top: 0.1rem;
            word-break: break-all;
        }
        .result-score {
            font-size: 0.75rem;
            color: #000000;
            margin-top: 0.2rem;
        }
        .no-results {
            margin-top: 1.5rem;
            color: #000000;
        }
        .chips {
            margin-top: 0.5rem;
            font-size: 0.8rem;
            color: #000000;
        }
        .chip-label {
            font-weight: 600;
            margin-right: 0.3rem;
        }
        .chip {
            display: inline-block;
            padding: 0.1rem 0.4rem;
            margin-right: 0.3rem;
            border-radius: 999px;
            border: 1px solid #374151;
        }
        a {
            color: inherit;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>ICS Search Engine</h1>
    <form method="get" action="/">
        <div class="search-bar">
            <input type="text" name="q" value="{{ query|e }}" placeholder="Search ICS websites..." autofocus>
            <button type="submit">Search</button>
        </div>
    </form>
    {% if query %}
        <div class="meta">
            {% if elapsed_ms is not none %}
                {{ result_count }} result(s) in {{ "%.1f"|format(elapsed_ms) }} ms
            {% else %}
                {{ result_count }} result(s)
            {% endif %}
        </div>
        <div class="chips">
            <span class="chip-label">Stems:</span>
            {% for s in stems %}
                <span class="chip">{{ s }}</span>
            {% endfor %}
            {% if bigrams %}
                <span class="chip-label" style="margin-left:0.6rem;">Bigrams:</span>
                {% for b in bigrams %}
                    <span class="chip">{{ b }}</span>
                {% endfor %}
            {% endif %}
        </div>
        {% if results %}
            <div class="result-list">
                {% for doc_id, score, url in results %}
                    <div class="result">
                        <a class="result-title" href="{{ url }}" target="_blank" rel="noopener noreferrer">
                            {{ url }}
                        </a>
                        <div class="result-url">{{ url }}</div>
                        <div class="result-score">doc {{ doc_id }} &middot; score {{ "%.4f"|format(score) }}</div>
                    </div>
                {% endfor %}
            </div>
        {% else %}
            <div class="no-results">No results found.</div>
        {% endif %}
    {% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    q = request.args.get("q", "").strip()
    if not q:
        return render_template_string(
            HTML_TEMPLATE,
            query="",
            stems=[],
            bigrams=[],
            results=[],
            result_count=0,
            elapsed_ms=None,
        )

    stems, bigrams = search.preprocess_query(q)
    all_terms = stems + bigrams
    bigram_set = set(bigrams)

    t0 = time.perf_counter()
    results_raw = search.ranked_search(
        all_terms,
        bigram_set,
        index_path,
        TERM_OFFSETS,
        NUM_DOCS,
        PAGERANK_SCORES,
        top_k=TOP_K_DEFAULT,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    results = []
    for doc_id, score in results_raw:
        url = DOCID_TO_URL.get(doc_id, "<missing URL>")
        results.append((doc_id, score, url))

    return render_template_string(
        HTML_TEMPLATE,
        query=q,
        stems=stems,
        bigrams=bigrams,
        results=results,
        result_count=len(results_raw),
        elapsed_ms=elapsed_ms,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)