"""Download the two p.2 reviewer-extension models (CPU only, no GPU).

    EleutherAI/pythia-2.8b  -> $MODELS_DIR/pythia-2.8b   (~5.7 GB fp16)
    microsoft/phi-1_5       -> $MODELS_DIR/phi-1_5       (~2.7 GB fp16)

Both are public base pretrains with UNTIED embeddings (verified in their HF
config.json: tie_word_embeddings = false).

DIRECT HTTP STREAMING, not huggingface_hub file transfer: in this environment
the library's LFS path (xet-bridge CDN) accepts the connection but transfers 0
bytes — observed live on pythia model.safetensors (8 min, .incomplete stayed at
0 bytes). Plain resolve-URL GETs work fine (~1.4 MB/s), so each whitelisted
file is streamed from https://huggingface.co/<repo>/resolve/main/<file> with
Range-based resume and a progress line every ~5 s (same pattern as the working
cmq_experiment/download_tinyllama_direct.py). list_repo_files still uses the
hub API, which works fine.

Whitelist: *.json / *.safetensors / *.bin / *.txt / *.model — config + tokenizer
+ weights; README/LICENSE/.gitattributes and other extras are skipped.
Idempotent: a file whose destination already exists with size > 0 is skipped;
interrupted downloads resume via .part files.
"""

import os
import sys
import time
from pathlib import Path

import requests
from huggingface_hub import list_repo_files


def models_dir():
    env = os.environ.get("MODELS_DIR")
    if env:
        return os.path.abspath(env)
    here = Path(__file__).resolve()                 # cross_model/
    parent = here.parents[2]                        # host tree (cmq_experiment/)
    cand = parent / "models"
    if cand.is_dir():
        return str(cand)
    raise RuntimeError("model directory not found. Set MODELS_DIR to the models folder.")


KEEP_EXT = (".json", ".safetensors", ".bin", ".txt", ".model")

REPOS = [
    ("EleutherAI/pythia-2.8b", "pythia-2.8b"),
    ("microsoft/phi-1_5",      "phi-1_5"),
]

CHUNK = 1024 * 1024          # 1 MB
MAX_RETRIES = 8


def fetch(repo, filename, dest_dir):
    """Stream one file from the HF resolve URL with resume support."""
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    dest = Path(dest_dir) / filename
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  {filename} already present ({dest.stat().st_size/1e6:.1f} MB), skipping", flush=True)
        return True
    part = dest.with_suffix(dest.suffix + ".part")

    head = requests.head(url, allow_redirects=True, timeout=30)
    total = int(head.headers.get("Content-Length") or 0) or None

    start = part.stat().st_size if part.exists() else 0
    if total and start >= total:
        print(f"  {filename}: .part already complete", flush=True)
        part.rename(dest)
        return True
    if start > 0:
        print(f"  {filename}: resuming from {start/1e6:.1f} MB", flush=True)

    completed = False
    for attempt in range(1, MAX_RETRIES + 1):
        headers = {"Range": f"bytes={start}-"} if start > 0 else {}
        try:
            with requests.get(url, stream=True, headers=headers,
                              allow_redirects=True, timeout=(30, 120)) as r:
                if start > 0 and r.status_code == 200:
                    start = 0
                    part.write_bytes(b"")
                r.raise_for_status()
                mode = "ab" if (start > 0 and r.status_code == 206) else "wb"
                if mode == "wb":
                    start = 0
                t0 = time.time()
                shown = 0.0
                with open(part, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        f.write(chunk)
                        start += len(chunk)
                        now = time.time()
                        if now - t0 >= 5 and now - shown >= 5:
                            shown = now
                            if total:
                                print(f"\r  {filename}: {100.0*start/total:5.1f}% "
                                      f"{start/1e6:8.1f}/{total/1e6:.0f} MB", end="", flush=True)
                            else:
                                print(f"\r  {filename}: {start/1e6:10.1f} MB (no total known)",
                                      end="", flush=True)
            print(flush=True)
            completed = True
            break
        except Exception as e:
            print(f"\n  [attempt {attempt}/{MAX_RETRIES}] {filename}: "
                  f"{type(e).__name__}: {e}", flush=True)
            print(f"    partial saved: {start/1e6:.1f} MB — will resume", flush=True)
            time.sleep(3 * attempt)

    ok = completed and (total is None or part.stat().st_size >= total)
    if ok:
        size = part.stat().st_size
        part.rename(dest)
        print(f"  {filename} DONE -> {size/1e6:.0f} MB", flush=True)
        return True
    print(f"  {filename} INCOMPLETE: {start/1e6:.1f}"
          + (f"/{total/1e6:.0f} MB" if total else " MB") + " (re-run to resume)", flush=True)
    return False


def main():
    dest_root = Path(models_dir())
    all_ok = True
    for repo, key in REPOS:
        dest = dest_root / key
        dest.mkdir(parents=True, exist_ok=True)
        files = [f for f in list_repo_files(repo) if f.endswith(KEEP_EXT)]
        print(f"=== {repo}: {len(files)} whitelisted files -> {dest}", flush=True)
        for fn in sorted(files):
            all_ok &= fetch(repo, fn, str(dest))
    if all_ok:
        print("DONE: both p.2 models in place.", flush=True)
    else:
        print("INCOMPLETE — re-run to resume (idempotent).", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
