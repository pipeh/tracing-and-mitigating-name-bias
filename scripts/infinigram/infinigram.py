import os
import csv
import time
import json
import random
import threading
import requests
import pandas as pd
from pathlib import Path
from dotenv import dotenv_values
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

index = "v4_dclm-baseline_llama"
API_URL = "https://api.infini-gram.io/"
ROOT = Path(__file__).resolve().parents[2]
ENV = dotenv_values(ROOT / ".env")
OUT_DIR = str(Path(ENV["DATA_DIR"]) / "infinigram" / index)
FIND_CACHE_PATH = os.path.join(OUT_DIR, "find_cache.csv")

TARGET_DOCS_PER_NAME = 1000
MAX_WORKERS = 16
REQUEST_SLEEP = 0.03
FLUSH_EVERY = 25

os.makedirs(OUT_DIR, exist_ok=True)

find_cache_lock = threading.Lock()

# HTTP helper
def post_with_retry(url, json_payload, max_retries=50, timeout=30):
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=json_payload, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
        except requests.exceptions.RequestException:
            if attempt == max_retries - 1:
                raise
        time.sleep(1)
    raise RuntimeError("Retry failure")

# Find-cache helpers
def load_find_cache():
    if not os.path.exists(FIND_CACHE_PATH):
        return {}
    df = pd.read_csv(FIND_CACHE_PATH)
    return {
        row["name"]: json.loads(row["segment_by_shard"])
        for _, row in df.iterrows()
    }

def save_find_cache(name, segments):
    with find_cache_lock:
        exists = os.path.exists(FIND_CACHE_PATH)
        row = pd.DataFrame([{
            "name": name,
            "segment_by_shard": json.dumps(segments)
        }])
        row.to_csv(
            FIND_CACHE_PATH,
            mode="a",
            header=not exists,
            index=False,
            quoting=csv.QUOTE_ALL
        )

# Rank pool
def build_rank_pool(segments, rng):
    pool = []
    for shard_idx, seg in enumerate(segments):
        if not seg:
            continue
        start, end = seg
        for r in range(start, end):
            pool.append((shard_idx, r))
    rng.shuffle(pool)
    return pool

# CSV helpers
def load_existing_docs(out_path):
    if not os.path.exists(out_path):
        return set(), 0
    df = pd.read_csv(out_path)
    return set(df["doc_ix"].tolist()), len(df)

def append_rows(out_path, rows):
    if not rows:
        return
    write_header = not os.path.exists(out_path)
    pd.DataFrame(rows).to_csv(
        out_path,
        mode="a",
        header=write_header,
        index=False,
        quoting=csv.QUOTE_ALL
    )

# Document sampling
def sample_unique_documents(
    index,
    name,
    segments,
    target_docs,
    already_seen,
    pbar,
    out_path,
    seed=42
):
    rng = random.Random(seed)
    rank_pool = build_rank_pool(segments, rng)

    buffer = []
    collected = 0

    for shard_idx, rank in rank_pool:
        if collected >= target_docs:
            break

        payload = {
            "index": index,
            "query_type": "get_doc_by_rank",
            "s": shard_idx,
            "rank": rank,
            "query": name
        }

        d = post_with_retry(API_URL, payload)
        doc_ix = d["doc_ix"]

        if doc_ix in already_seen:
            continue

        already_seen.add(doc_ix)

        text = " ".join(span[0] for span in d["spans"])
        buffer.append({
            "doc_ix": doc_ix,
            "doc_len": d["doc_len"],
            "disp_len": d["disp_len"],
            "text": text,
        })

        collected += 1
        pbar.update(1)

        if len(buffer) >= FLUSH_EVERY:
            append_rows(out_path, buffer)
            buffer.clear()

        time.sleep(REQUEST_SLEEP)

    # final flush
    append_rows(out_path, buffer)

    return collected


# Main per-name logic
def process_name(name, index, find_cache, position):
    out_path = os.path.join(OUT_DIR, f"{name}.csv")

    seen_doc_ix, current_count = load_existing_docs(out_path)

    if current_count >= TARGET_DOCS_PER_NAME:
        return name, current_count

    remaining = TARGET_DOCS_PER_NAME - current_count

    if name in find_cache:
        segments = find_cache[name]
    else:
        payload_find = {
            "index": index,
            "query_type": "find",
            "query": name
        }
        result_find = post_with_retry(API_URL, payload_find)
        if "error" in result_find:
            raise RuntimeError(result_find["error"])

        segments = result_find.get("segment_by_shard", [])
        save_find_cache(name, segments)
        find_cache[name] = segments

    seed = hash(name) & 0xffffffff

    with tqdm(
        total=TARGET_DOCS_PER_NAME,
        initial=current_count,
        desc=name,
        position=position,
        leave=False,
        dynamic_ncols=True,
        smoothing=0.1
    ) as pbar:

        added = sample_unique_documents(
            index=index,
            name=name,
            segments=segments,
            target_docs=remaining,
            already_seen=seen_doc_ix,
            pbar=pbar,
            out_path=out_path,
            seed=seed
        )

    return name, current_count + added

# Load names
with open(ROOT / "external/data/fb_names.json", "r") as f:
    names_dict = json.load(f)

names = []
for k in names_dict:
    for c in names_dict[k]:
        names.extend(names_dict[k][c])

names = sorted(set(names))
print(f"Total names: {len(names)}")

# Load find cache
find_cache = load_find_cache()
print(f"Find cache loaded: {len(find_cache)} names")

# Parallel execution
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(process_name, n, index, find_cache, i % MAX_WORKERS): n
        for i, n in enumerate(names)
    }

    for future in as_completed(futures):
        name = futures[future]
        try:
            name, count = future.result()
            print(f"[DONE] {name}: {count}")
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
