import argparse
import json
import re
import collections
import hashlib
import time
from pathlib import Path
from urllib.parse import urljoin, urldefrag
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from collections import Counter
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

token_regex = re.compile(r"[A-Za-z0-9]+")

try:
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
except Exception:
    stemmer = None

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

def load_json_page(json_path):
    try:
        data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
        url = data.get("url", str(json_path))
        html = data.get("content", "")
        if not isinstance(html, str):
            html = str(html)
        return url, html
    except Exception:
        return str(json_path), ""

def collect_json_files(folder):
    return list(folder.rglob("*.json"))

def extract_visible_text(html_content):
    try:
        soup = BeautifulSoup(html_content or "", "lxml")
    except Exception:
        soup = BeautifulSoup(html_content or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return soup.get_text(separator=" ", strip=True)

# resolve relative links and drop fragments & returns none on invalid urls
def normalize_url(base_url, href):
    try:
        full = urljoin(str(base_url), str(href))
        full, _ = urldefrag(full)
        return full
    except Exception:
        return None


# simhash / near-duplicate helpers --------------------------------------------------

def compute_simhash(tokens, hash_bits=64):
    if not tokens:
        return 0

    counts = Counter(tokens)

    v = [0] * hash_bits
    for term, w in counts.items():
        h = int(hashlib.md5(term.encode("utf-8", errors="ignore")).hexdigest(), 16)
        for i in range(hash_bits):
            bit = (h >> i) & 1
            if bit:
                v[i] += w
            else:
                v[i] -= w

    sim = 0
    for i in range(hash_bits):
        if v[i] > 0:
            sim |= (1 << i)
    return sim

# count differing bits between two integers
def hamming_distance(x, y):
    return (x ^ y).bit_count()

# extract one band of bits from the signature
def get_simhash_band(signature, band_idx, bits_per_band=16):
    shift = band_idx * bits_per_band
    mask = (1 << bits_per_band) - 1
    return (signature >> shift) & mask


# term info (content + positions + bigrams) ------------------------------------------------------

def extract_term_info_from_soup(soup):
    # base visible text for positions and bigrams
    body_text = soup.get_text(separator=" ", strip=True)
    tokens = tokenize_text(body_text)
    stemmed_tokens = [stem_token(t) for t in tokens]

    term_positions = {}
    for idx, tok in enumerate(stemmed_tokens):
        if not tok:
            continue
        term_positions.setdefault(tok, []).append(idx)

    term_tf = {term: len(pos_list) for term, pos_list in term_positions.items()}

    # field weighting -- title, headings, bold
    extra_weight = collections.Counter()

    def add_weight_from_text(text, weight):
        for tok in tokenize_text(text):
            stem = stem_token(tok)
            if stem:
                extra_weight[stem] += weight

    # title
    if soup.title and soup.title.string:
        add_weight_from_text(soup.title.get_text(separator=" ", strip=True), weight=3)

    # headings
    for h in soup.find_all(["h1", "h2", "h3"]):
        add_weight_from_text(h.get_text(separator=" ", strip=True), weight=2)

    # bold / strong
    for b in soup.find_all(["b", "strong"]):
        add_weight_from_text(b.get_text(separator=" ", strip=True), weight=2)

    # apply extra weights
    for term, w in extra_weight.items():
        term_tf[term] = term_tf.get(term, 0) + w

    # bigram terms (from stemmed tokens)
    for i in range(len(stemmed_tokens) - 1):
        t1 = stemmed_tokens[i]
        t2 = stemmed_tokens[i + 1]
        if not t1 or not t2:
            continue
        bigram_term = f"{t1}_{t2}"
        # position = start index of the bigram
        term_positions.setdefault(bigram_term, []).append(i)
        term_tf[bigram_term] = term_tf.get(bigram_term, 0) + 1

    return term_tf, term_positions

# anchor text -> target pages -----------------------------------

def extract_anchor_terms_for_targets_from_soup(soup, page_url, url_to_docid_map):
    anchor_terms_for_targets = {}

    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        href = href.strip()
        if not href or href.startswith("#"):
            continue
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue

        target_url = normalize_url(page_url, href)
        if not target_url:
            continue
        if target_url not in url_to_docid_map:
            continue

        target_doc_id = url_to_docid_map[target_url]
        anchor_text = a.get_text(separator=" ", strip=True)
        if not anchor_text:
            continue

        tokens = tokenize_text(anchor_text)
        if not tokens:
            continue

        term_counts = anchor_terms_for_targets.setdefault(target_doc_id, {})
        for tok in tokens:
            stem = stem_token(tok)
            if stem:
                term_counts[stem] = term_counts.get(stem, 0) + 1

    return anchor_terms_for_targets


# partial index writing / merge -----------------------------------------------------

def write_partial_index(part_number, inverted_index, output_folder):
    part_path = output_folder / f"partial_{part_number:03d}.jsonl"
    with part_path.open("w", encoding="utf-8") as f:
        for term in sorted(inverted_index.keys()):
            postings = []
            for doc_id, (tf, pos_list) in inverted_index[term].items():
                postings.append([doc_id, tf, pos_list])
            postings.sort(key=lambda x: x[0])
            f.write(json.dumps({"term": term, "postings": postings}) + "\n")
    return part_path

def merge_partial_indexes(part_paths, final_path):
    files = [p.open("r", encoding="utf-8") for p in part_paths]
    try:
        cursors = []
        for idx, fh in enumerate(files):
            line = fh.readline()
            if line:
                obj = json.loads(line)
                cursors.append((obj["term"], obj, idx))

        with final_path.open("w", encoding="utf-8") as out:
            while cursors:
                cursors.sort(key=lambda x: x[0])
                current_term = cursors[0][0]
                combined = {}
                refill = []

                while cursors and cursors[0][0] == current_term:
                    _, entry, file_index = cursors.pop(0)
                    for doc_id, tf, pos_list in entry["postings"]:
                        if doc_id not in combined:
                            combined[doc_id] = [tf, list(pos_list)]
                        else:
                            combined[doc_id][0] += tf
                            combined[doc_id][1].extend(pos_list)

                    nxt = files[file_index].readline()
                    if nxt:
                        nxt_obj = json.loads(nxt)
                        refill.append((nxt_obj["term"], nxt_obj, file_index))

                cursors.extend(refill)
                postings = []
                for doc_id, (tf, pos_list) in combined.items():
                    postings.append([doc_id, tf, pos_list])
                postings.sort(key=lambda x: x[0])
                out.write(json.dumps({"term": current_term, "postings": postings}) + "\n")
    finally:
        for fh in files:
            fh.close()


# duplicate detection (exact + near) --------------------

def build_url_to_docid_and_duplicates(data_folder):
    json_files = collect_json_files(data_folder)

    url_to_docid_map = {}
    url_to_path_map = {}

    # exact duplicates
    seen_hashes = {}

    # near-duplicates (simhash)
    num_bands = 4
    bits_per_band = 16
    simhash_buckets = [collections.defaultdict(list) for _ in range(num_bands)] 
    sigs = {}

    duplicates = []

    for json_file in json_files:
        url, html = load_json_page(json_file)
        visible_text = extract_visible_text(html)
        content_bytes = visible_text.encode("utf-8", errors="ignore")

        # exact duplicate detection (md5)
        content_hash = hashlib.md5(content_bytes).hexdigest()
        if content_hash in seen_hashes:
            canonical_url = seen_hashes[content_hash]
            duplicates.append((url, canonical_url))
            continue

        # near-duplicate detection (simhash)
        tokens = tokenize_text(visible_text)
        stems = [stem_token(t) for t in tokens if t]
        sig = compute_simhash(stems)

        near_dup_main_url = None
        if sig != 0:  # if we have some text
            for band_idx in range(num_bands):
                band_val = get_simhash_band(sig, band_idx, bits_per_band)
                bucket = simhash_buckets[band_idx].get(band_val, [])
                for cand_url, cand_sig in bucket:
                    if hamming_distance(sig, cand_sig) <= 3:
                        near_dup_main_url = cand_url
                        break
                if near_dup_main_url is not None:
                    break

        if near_dup_main_url is not None:
            # near-duplicate of existing page
            duplicates.append((url, near_dup_main_url))
            continue

        # new page
        seen_hashes[content_hash] = url
        sigs[url] = sig
        if sig != 0:
            for band_idx in range(num_bands):
                band_val = get_simhash_band(sig, band_idx, bits_per_band)
                simhash_buckets[band_idx][band_val].append((url, sig))

        # assign doc_id and remember file path
        doc_id = len(url_to_docid_map)
        url_to_docid_map[url] = doc_id
        url_to_path_map[url] = json_file

    return url_to_docid_map, url_to_path_map, duplicates


# -------------------- pagerank --------------------

def compute_pagerank(num_docs, out_links, max_iters=40, d=0.85):
    if num_docs == 0:
        return []

    pr = [1.0 / num_docs] * num_docs
    out_deg = [0] * num_docs
    for doc_id in range(num_docs):
        out_deg[doc_id] = len(out_links.get(doc_id, []))

    for _ in range(max_iters):
        new_pr = [0.0] * num_docs
        dangling_sum = 0.0

        for i in range(num_docs):
            if out_deg[i] == 0:
                dangling_sum += pr[i]
            else:
                share = pr[i] / out_deg[i]
                for j in out_links[i]:
                    new_pr[j] += share

        for j in range(num_docs):
            new_pr[j] = d * (new_pr[j] + dangling_sum / num_docs) + (1.0 - d) / num_docs

        pr = new_pr

    max_pr = max(pr) if pr else 1.0
    if max_pr > 0:
        pr = [p / max_pr for p in pr]
    return pr

# -------------------- main index build --------------------

def build_external_index(data_folder, output_folder, memory_limit_mb):
    memory_limit_bytes = memory_limit_mb * 1024 * 1024

    url_to_docid_map, url_to_path_map, duplicates = build_url_to_docid_and_duplicates(data_folder)

    inverted_index = collections.defaultdict(lambda: collections.defaultdict(tuple))
    document_table = [None] * len(url_to_docid_map)
    partial_paths = []
    estimated_bytes = 0

    out_links = {doc_id: set() for doc_id in range(len(url_to_docid_map))}

    # index content + anchors
    for url, doc_id in url_to_docid_map.items():
        json_file = url_to_path_map[url]
        _, html = load_json_page(json_file)

        try:
            soup = BeautifulSoup(html or "", "lxml")
        except Exception:
            soup = BeautifulSoup(html or "", "html.parser")

        # ---------------- content terms + positions ----------------
        term_tf, term_positions = extract_term_info_from_soup(soup)
        doc_length = sum(term_tf.values())
        document_table[doc_id] = (url, doc_length)

        # content contributions
        for term, tf in term_tf.items():
            pos_list = term_positions.get(term, [])
            existing = inverted_index[term].get(doc_id)
            if existing is None:
                inverted_index[term][doc_id] = (tf, pos_list)
            else:
                old_tf, old_pos = existing
                inverted_index[term][doc_id] = (old_tf + tf, old_pos + pos_list)
            estimated_bytes += len(term) + 12 + len(pos_list) * 4

        # ---------------- anchor contributions to target pages ----------------
        anchor_terms_for_targets = extract_anchor_terms_for_targets_from_soup(soup, url, url_to_docid_map)
        for target_doc_id, anchor_tf_map in anchor_terms_for_targets.items():
            for term, anchor_tf in anchor_tf_map.items():
                existing = inverted_index[term].get(target_doc_id)
                if existing is None:
                    inverted_index[term][target_doc_id] = (anchor_tf, [])
                else:
                    old_tf, old_pos = existing
                    inverted_index[term][target_doc_id] = (old_tf + anchor_tf, old_pos)
                estimated_bytes += len(term) + 12

        # ---------------- build out-links for pagerank ----------------
        for a in soup.find_all("a"):
            href = a.get("href")
            if not href:
                continue
            href = href.strip()
            if not href or href.startswith("#"):
                continue
            if href.startswith("mailto:") or href.startswith("javascript:"):
                continue
            target_url = normalize_url(url, href)
            if not target_url:
                continue
            if target_url in url_to_docid_map:
                target_id = url_to_docid_map[target_url]
                if target_id != doc_id:
                    out_links[doc_id].add(target_id)

        # spill partial index if over memory
        if estimated_bytes >= memory_limit_bytes:
            part_path = write_partial_index(len(partial_paths), inverted_index, output_folder)
            partial_paths.append(part_path)
            inverted_index.clear()
            estimated_bytes = 0

    # final spill
    if inverted_index:
        part_path = write_partial_index(len(partial_paths), inverted_index, output_folder)
        partial_paths.append(part_path)

    final_index_path = output_folder / "index_merged.jsonl"
    merge_partial_indexes(partial_paths, final_index_path)

    # compute pagerank and save to file
    num_docs = len(url_to_docid_map)
    pr_scores = compute_pagerank(num_docs, out_links)
    pr_path = output_folder / "pagerank.json"
    with pr_path.open("w", encoding="utf-8") as f:
        json.dump(pr_scores, f)

    # write duplicates file
    if duplicates:
        dup_path = output_folder / "duplicates.tsv"
        with dup_path.open("w", encoding="utf-8") as f:
            f.write("duplicate_url\tcanonical_url\n")
            for dup_url, canonical_url in duplicates:
                f.write(f"{dup_url}\t{canonical_url}\n")

    return final_index_path, document_table

# -------------------- doc table + analytics --------------------

def save_doc_table(document_table, output_folder):
    with (output_folder / "doc_table.tsv").open("w", encoding="utf-8") as f:
        for doc_id, entry in enumerate(document_table):
            if entry is None:
                continue
            url, length = entry
            f.write(f"{doc_id}\t{url}\t{length}\n")


def compute_analytics(index_file, document_table, output_folder):
    unique_terms = sum(1 for _ in index_file.open("r", encoding="utf-8"))
    index_kb = index_file.stat().st_size // 1024
    with (output_folder / "analytics.csv").open("w", encoding="utf-8") as f:
        f.write("doc_count,unique_terms,index_size_kb\n")
        f.write(f"{len(document_table)},{unique_terms},{index_kb}\n")


def main():
    parser = argparse.ArgumentParser(description="ICS Search Engine Indexer")
    parser.add_argument("--data-folder", type=Path, required=True,help="Root folder with JSON files")
    parser.add_argument("--output-folder", type=Path, required=True,help="Where to write index_merged.jsonl and doc_table.tsv")
    parser.add_argument("--memory-limit", type=int, default=256,help="Approximate memory limit before spilling partial index")
    args = parser.parse_args()
    output_folder = args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    index_file, document_table = build_external_index(args.data_folder, output_folder, args.memory_limit)
    save_doc_table(document_table, output_folder)
    compute_analytics(index_file, document_table, output_folder)
    elapsed = time.perf_counter() - start_time

    print("\n=== Successful Index Build===")
    print(f"Indexed documents : {len(document_table)}")
    print(f"Index file        : {index_file}")
    print(f"Output folder     : {output_folder.resolve()}")
    print(f"Elapsed time      : {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()