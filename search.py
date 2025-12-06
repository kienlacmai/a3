import argparse
import json
import math
import re
import time
from pathlib import Path

# -------------------- tokenization / stemming --------------------

token_regex = re.compile(r"[A-Za-z0-9]+")

try:
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
except Exception:
    stemmer = None

#  common function words to skip
STOPWORDS = {
    "the", "of", "and", "to", "a", "an", "in", "for", "on", "at",
    "by", "with", "is", "are", "be", "from", "that", "this", "it",
    "as", "into", "up", "down", "over", "under"
}

def tokenize_text(text):
    return [m.group(0).lower() for m in token_regex.finditer(text or "")]

def stem_token(word):
    word = word.lower()
    if stemmer:
        try:
            return stemmer.stem(word)
        except Exception:
            pass
    for ending in ("ing", "ed", "s"):
        if word.endswith(ending) and len(word) > len(ending) + 2:
            return word[:-len(ending)]
    return word


def preprocess_query(raw_query):
    # tokenize and drop stopwords up front
    tokens = tokenize_text(raw_query)
    filtered = [t for t in tokens if t not in STOPWORDS]

    # stem remaining tokens
    stems = [stem_token(t) for t in filtered if t]

    # build bigrams over stemmed tokens (no stopwords inside)
    bigrams = []
    for i in range(len(stems) - 1):
        if stems[i] and stems[i + 1]:
            bigrams.append(f"{stems[i]}_{stems[i+1]}")
    return stems, bigrams



# -------------------- doc table --------------------

def load_doc_table(doc_table_path):
    docid_to_url = {}
    with doc_table_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                doc_id = int(parts[0])
            except ValueError:
                continue
            url = parts[1]
            docid_to_url[doc_id] = url
    return docid_to_url


# -------------------- pagerank loader --------------------

def load_pagerank(pr_path, num_docs):
    if not pr_path.exists():
        return [0.0] * num_docs
    try:
        with pr_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) >= num_docs:
            return [float(x) for x in data[:num_docs]]
        else:
            return [0.0] * num_docs
    except Exception:
        return [0.0] * num_docs


# -------------------- lexicon --------------------

def build_lexicon(index_path, lexicon_path):
    term_to_offset = {}
    with index_path.open("r", encoding="utf-8") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            obj = json.loads(line)
            term = obj["term"]
            term_to_offset[term] = offset
    with lexicon_path.open("w", encoding="utf-8") as f:
        json.dump(term_to_offset, f)
    return term_to_offset


def load_lexicon(index_path, lexicon_path):
    if lexicon_path.exists():
        with lexicon_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return build_lexicon(index_path, lexicon_path)


# -------------------- load postings (with positions) --------------------

def load_postings_for_terms(index_path, term_offsets, terms):
    postings = {}
    with index_path.open("r", encoding="utf-8") as f:
        for term in terms:
            offset = term_offsets.get(term)
            if offset is None:
                postings[term] = []
                continue
            f.seek(offset)
            line = f.readline()
            if not line:
                postings[term] = []
                continue
            obj = json.loads(line)
            term_postings = []
            for doc, tf, positions in obj["postings"]:
                term_postings.append((int(doc), int(tf), positions))
            postings[term] = term_postings
    return postings


# -------------------- ranked search with proximity + pagerank --------------------

def ranked_search(all_terms,bigram_terms,index_path,term_offsets,num_docs,pagerank_scores,pr_weight=0.3,top_k=5):

    # load postings for every term in the query (unigrams + bigrams)
    postings_dict = load_postings_for_terms(index_path, term_offsets, all_terms)

    # use only unigrams for and/or candidate selection
    unigram_terms = [t for t in all_terms if t not in bigram_terms]
    base_terms = unigram_terms or all_terms

    # ---- choose rarest terms first for candidate set ----
    term_postings_pairs = [
        (t, postings_dict[t])
        for t in base_terms
        if postings_dict.get(t)
    ]
    if not term_postings_pairs:
        return []

    # sort so that we intersect shortest postings first
    term_postings_pairs.sort(key=lambda p: len(p[1]))
    nonempty_postings = [p[1] for p in term_postings_pairs]

    # and intersection
    candidate_docs = set(doc_id for doc_id, _, _ in nonempty_postings[0])
    for plist in nonempty_postings[1:]:
        candidate_docs &= {doc_id for doc_id, _, _ in plist}
        if not candidate_docs:
            break

    # or fallback
    if not candidate_docs:
        candidate_docs = set()
        for _, plist in term_postings_pairs:
            candidate_docs |= {doc_id for doc_id, _, _ in plist}
        if not candidate_docs:
            return []

    # ---- tf-idf cosine scoring with bigram boost ----
    scores = {}
    doc_norm_sq = {}
    query_norm_sq = 0.0

    for term in all_terms:
        postings = postings_dict.get(term)
        if not postings:
            continue

        df = len(postings)
        if df == 0:
            continue

        idf = math.log((num_docs + 1) / (df + 1)) + 1.0

        # bigram boost
        if term in bigram_terms:
            w_qt = idf * 2.0
        else:
            w_qt = idf

        query_norm_sq += w_qt * w_qt

        for doc_id, tf, _ in postings:
            if doc_id not in candidate_docs:
                continue
            w_dt = tf * idf
            scores[doc_id] = scores.get(doc_id, 0.0) + w_qt * w_dt
            doc_norm_sq[doc_id] = doc_norm_sq.get(doc_id, 0.0) + w_dt * w_dt

    if not scores:
        return []

    query_norm = math.sqrt(query_norm_sq) or 1.0
    for doc_id in list(scores.keys()):
        doc_norm = math.sqrt(doc_norm_sq.get(doc_id, 0.0)) or 1.0
        scores[doc_id] = scores[doc_id] / (query_norm * doc_norm)

    # ---- proximity bonus using positions, but avoid rescanning huge postings ----
    if len(unigram_terms) >= 2:
        # precompute positions per term for just the candidate docs
        positions_by_term = {}
        for term in unigram_terms:
            term_positions_for_docs = {}
            postings = postings_dict.get(term, [])
            for d, tf, pos_list in postings:
                if d in candidate_docs:
                    term_positions_for_docs[d] = pos_list
            positions_by_term[term] = term_positions_for_docs

        for doc_id in list(scores.keys()):
            term_positions_lists = []
            for term in unigram_terms:
                pos_list = positions_by_term.get(term, {}).get(doc_id)
                if not pos_list:
                    term_positions_lists = []
                    break
                term_positions_lists.append(pos_list)

            if not term_positions_lists:
                continue

            # compute minimal pairwise distance (simple but on small sets now)
            min_dist = 10**9
            for i in range(len(term_positions_lists)):
                for j in range(i + 1, len(term_positions_lists)):
                    for p1 in term_positions_lists[i]:
                        for p2 in term_positions_lists[j]:
                            dist = abs(p1 - p2)
                            if dist < min_dist:
                                min_dist = dist

            if min_dist < 6:
                scores[doc_id] *= 1.25
            elif min_dist < 12:
                scores[doc_id] *= 1.10

    # ---- mix in pagerank (pr already normalized to [0,1]) ----
    if pagerank_scores:
        for doc_id in list(scores.keys()):
            pr = 0.0
            if 0 <= doc_id < len(pagerank_scores):
                pr = pagerank_scores[doc_id]
            scores[doc_id] = scores[doc_id] + pr_weight * pr

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked[:top_k]

# -------------------- print results --------------------

def print_results(raw_query, stems, bigrams, results, docid_to_url, elapsed_ms):
    print(f"\nQuery: {raw_query!r}")
    print(f"Processed stems  : {stems}")
    print(f"Processed bigrams: {bigrams}")
    print(f"Results in {elapsed_ms:.1f} ms\n")
    if not results:
        print("No results found.")
        return
    for rank, (doc_id, score) in enumerate(results, start=1):
        url = docid_to_url.get(doc_id, "<missing URL>")
        print(f"{rank}. [doc {doc_id}] score={score:.4f}")
        print(f"   {url}")

# -------------------- main loop --------------------

def main():
    parser = argparse.ArgumentParser(description="CS 121 Search Engine")
    parser.add_argument("--index-folder", type=Path, required=True,help="Folder with index_merged.jsonl and doc_table.tsv")
    parser.add_argument("--index-file-name", type=str, default="index_merged.jsonl")
    parser.add_argument("--doc-table-name", type=str, default="doc_table.tsv")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    index_path = args.index_folder / args.index_file_name
    doc_table_path = args.index_folder / args.doc_table_name

    print("Loading document table...")
    docid_to_url = load_doc_table(doc_table_path)
    num_docs = len(docid_to_url)
    print(f"Loaded {num_docs} documents.")

    # load pagerank scores
    pr_path = args.index_folder / "pagerank.json"
    pagerank_scores = load_pagerank(pr_path, num_docs)
    print("Loaded PageRank scores.\n")

    lexicon_path = index_path.with_suffix(".lexicon.json")
    print("Building or loading lexicon...")
    term_offsets = load_lexicon(index_path, lexicon_path)
    print(f"Lexicon loaded for {len(term_offsets)} terms.\n")

    print("=== Ranked Search ===")
    print("Type a query, or 'exit' to quit.\n")

    while True:
        try:
            raw = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw or raw.lower() == "exit":
            break

        stems, bigrams = preprocess_query(raw)
        all_terms = stems + bigrams

        t0 = time.perf_counter()
        results = ranked_search(all_terms,set(bigrams),index_path,term_offsets,num_docs,pagerank_scores,top_k=args.top_k,)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        print_results(raw, stems, bigrams, results, docid_to_url, elapsed_ms)
        print()

if __name__ == "__main__":
    main()
