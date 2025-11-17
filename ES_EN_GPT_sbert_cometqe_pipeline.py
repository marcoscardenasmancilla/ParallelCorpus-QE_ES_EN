#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ES–EN Parallel Corpus Pipeline v1.2-safe (Batch + COMET-QE + QE_mix)
-------------------------------------------------------------------------------
- Aligns ES/EN by sent_id from CoNLL-U files in a common directory.
- Computes per sentence:
    • COMET-QE (no human reference): estimated quality source→hypothesis.
    • S-BERT ES↔EN (cosine): multilingual semantic adequacy (chunked + CPU-first).
    • Sanity signals: length ratio, digits, punctuation Jaccard.
    • Optional heuristic: character n-gram cosine (char-N).
- Creates composite metric QE_mix (min–max normalized within the batch):
    QE_mix = w_comet * COMET-QE_norm + w_sbert * SBERT_norm
- Exports CSVs, HTML with plots, and a per-pair summary.

ROBUSTNESS (anti-crash):
- CPU-first by default (COMET_QE_DEVICE="cpu"), small batch (8), OMP/MKL thread control.
- Automatic GPU detection if available; retries on CPU on OOM.
- S-BERT uses batch/chunk encoding to avoid memory spikes.
- NO BLEU/chrF (no valid monolingual reference).

NOTE:
- Requires: pandas, numpy, matplotlib, sentence-transformers, unbabel-comet.
"""

# ======= Safe-mode env vars (before importing heavy libraries) =======
import os as _os
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# (Optional) If you use PyTorch CPU and want to fix threads:
try:
    import torch as _torch
    if not _torch.cuda.is_available():
        _torch.set_num_threads(1)
except Exception:
    pass

# ======================================================================

import re, math, unicodedata, glob
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os
import sys
from pathlib import Path
import warnings
import traceback

# ========== CONFIG ==========
CONFIG = {
    # --- Batch: common directory with all parallel .conllu files ---
    "COMMON_SOURCE_DIR": r" ",

    # --- Root output directory; subfolders per pair will be created ---
    "COMMON_OUTPUTS_DIR": r" ",

    # (Optional) If you want to process ONLY certain sentence ids (e.g. "1,3,5-8")
    "IDS": None,

    # -------- Sanity signals --------
    "LEN_LO": 0.60,
    "LEN_HI": 1.60,

    # -------- S-BERT (ES↔EN adequacy) --------
    "SBERT_ENABLE": True,
    "SBERT_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "SBERT_DEVICE": "cpu",        # "cuda" if you have a stable GPU
    "SBERT_BATCH_SIZE": 64,       # chunk size for encode
    "SBERT_LOW": 0.70,            # alert flag threshold

    # -------- COMET-QE (reference-less quality) --------
    "COMET_QE_ENABLE": True,
    # Lighter/more stable by default; you can change to "Unbabel/wmt22-cometkiwi-da"
    "COMET_QE_MODEL": "Unbabel/wmt20-comet-qe-da",
    "COMET_QE_DEVICE": "cpu",     # "cuda:0" if you have a stable GPU
    "COMET_QE_BATCH_SIZE": 8,     # conservative to avoid OOM

    # -------- Composite metric --------
    "QE_MIX": True,
    "QE_MIX_WEIGHTS": {"comet": 0.6, "sbert": 0.4},
    "QE_MIX_LOW": 0.35,           # alert threshold

    # -------- Optional surface heuristic --------
    "SURFACE_SIM_ENABLE": True,
    "CHAR_N": 3,                  # n for char-ngrams
    "COSINE_CHARN_LOW": 0.15      # alert flag (heuristic)
}

# ========== Optional libraries ==========
# S-BERT (robust loading)
_HAS_SBERT = False
_SbertModel = None
try:
    if CONFIG["SBERT_ENABLE"]:
        from sentence_transformers import SentenceTransformer
        # Device control: some models respect device in encode; we will pass it there.
        _SbertModel = SentenceTransformer(CONFIG["SBERT_MODEL"])
        _HAS_SBERT = True
except Exception as e:
    _HAS_SBERT = False
    _SbertModel = None
    print(f"[WARN] S-BERT not available ({e}). Set SBERT_ENABLE=False or install sentence-transformers.")

# COMET-QE (robust loading)
_HAS_COMET_QE = False
_COMET_QE_MODEL_OBJ = None
try:
    if CONFIG.get("COMET_QE_ENABLE", False):
        from comet import download_model, load_from_checkpoint
        ckpt_path = download_model(CONFIG["COMET_QE_MODEL"])
        _COMET_QE_MODEL_OBJ = load_from_checkpoint(ckpt_path)
        _HAS_COMET_QE = True
except Exception as e:
    print(f"[WARN] COMET-QE not available: {e}")
    _HAS_COMET_QE = False
    _COMET_QE_MODEL_OBJ = None

# ========== Utilities ==========


def ensure_dir(p):
    """
    Ensure directory exists and return absolute path.
    Raises ValueError if p is None or empty.
    """
    if p is None:
        raise ValueError("ensure_dir: path is None. Check CONFIG keys (COMMON_OUTPUTS_DIR, etc.).")
    # allow passing Path or str
    ppath = Path(p).expanduser().resolve()
    ppath.mkdir(parents=True, exist_ok=True)
    return str(ppath)


def strip_diacritics(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))


def extract_numbers(s: str):
    return set(re.findall(r"\d+(?:[.,]\d+)?", s)) if s is not None else set()


def punctuation_set(s: str):
    return set(re.findall(r"[^\w\s]", s)) if s is not None else set()


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else (1.0 if not a and not b else 0.0)


def char_ngrams(s: str, n: int = 3):
    if s is None:
        return []
    s = strip_diacritics(s.lower())
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [s[i:i + n] for i in range(len(s) - n + 1)] if len(s) >= n else []


def cosine_from_ngrams(a: str, b: str, n: int = 3) -> float:
    na, nb = Counter(char_ngrams(a, n)), Counter(char_ngrams(b, n))
    if not na or not nb:
        return 0.0
    dot = sum(na[g] * nb.get(g, 0) for g in na)
    norm_a = math.sqrt(sum(v * v for v in na.values()))
    norm_b = math.sqrt(sum(v * v for v in nb.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _batch_iter(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ========== CoNLL-U parsing ==========
def parse_conllu(path: str) -> Dict[str, Any]:
    data = {}
    sent_id = None
    sent_text = None
    rows = []

    def flush():
        nonlocal sent_id, sent_text, rows
        if sent_id:
            lemmas = [r[2] for r in rows if len(r) >= 10 and '-' not in r[0] and '.' not in r[0]]
            upos = [r[3] for r in rows if len(r) >= 10 and '-' not in r[0] and '.' not in r[0]]
            data[sent_id] = {"text": sent_text or "", "tokens": rows[:], "lemmas": lemmas, "upos": upos}
        sent_id, sent_text, rows = None, None, []

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    flush()
                    continue
                if line.startswith("# sent_id"):
                    sent_id = line.split("=", 1)[1].strip()
                elif line.startswith("# text"):
                    sent_text = line.split("=", 1)[1].strip()
                elif not line.startswith("#"):
                    rows.append(line.split("\t")[:10])
        if sent_id:
            flush()
    except Exception as e:
        print(f"[ERR] parse_conllu failed on {path}: {e}")
    return data


# ========== S-BERT ES↔EN ==========
def sbert_batch_cosine(es_texts: List[str], en_texts: List[str]) -> List[Optional[float]]:
    if (not _HAS_SBERT) or (_SbertModel is None) or (not es_texts):
        return [None] * len(es_texts)
    # Chunked encoding to avoid OOM; encode on configured device
    device = CONFIG.get("SBERT_DEVICE", "cpu")
    bs = max(1, int(CONFIG.get("SBERT_BATCH_SIZE", 64)))

    # Encode ES
    es_vecs = []
    for chunk in _batch_iter(es_texts, bs):
        es_emb = _SbertModel.encode(chunk, convert_to_tensor=True, normalize_embeddings=True, device=device)
        es_vecs.append(es_emb)
    import torch
    es_all = torch.cat(es_vecs, dim=0)

    # Encode EN
    en_vecs = []
    for chunk in _batch_iter(en_texts, bs):
        en_emb = _SbertModel.encode(chunk, convert_to_tensor=True, normalize_embeddings=True, device=device)
        en_vecs.append(en_emb)
    en_all = torch.cat(en_vecs, dim=0)

    sims = (es_all * en_all).sum(dim=1).detach().cpu().numpy()
    return sims.tolist()


# ========== COMET-QE ==========
def _gpu_available(device_str: str) -> bool:
    try:
        import torch
        return device_str.startswith("cuda") and torch.cuda.is_available()
    except Exception:
        return False


def comet_qe_score_batch(es_texts, en_texts, device="cpu", batch_size=8):
    """Returns a robust list of COMET-QE scores; retries on CPU with a smaller batch on OOM."""
    if not _HAS_COMET_QE or _COMET_QE_MODEL_OBJ is None or not es_texts:
        return [float("nan")] * (len(es_texts) if es_texts else 0)

    data = [{"src": (s or ""), "mt": (t or "")} for s, t in zip(es_texts, en_texts)]
    use_gpu = _gpu_available(str(device))
    bsz = max(1, int(batch_size))

    def _predict(_bsz, _gpus):
        return _COMET_QE_MODEL_OBJ.predict(
            data, batch_size=_bsz, gpus=_gpus, progress_bar=False
        )

    try:
        out = _predict(bsz, 1 if use_gpu else 0)
        if isinstance(out, dict) and "scores" in out:
            return [float(x) for x in out["scores"]]
        if isinstance(out, dict) and "segments" in out:
            return [float(seg.get("score", float("nan"))) for seg in out["segments"]]
        return [float("nan")] * len(data)
    except RuntimeError as e:
        # Typical OOM -> retry on CPU with minimal batch
        try:
            out = _predict(4, 0)
            if "scores" in out:
                return [float(x) for x in out["scores"]]
            if "segments" in out:
                return [float(seg.get("score", float("nan"))) for seg in out["segments"]]
        except Exception:
            pass
        print(f"[WARN] COMET-QE OOM/Runtime: {e}")
        return [float("nan")] * len(es_texts)
    except Exception as e:
        print(f"[WARN] COMET-QE predict failed: {e}")
        return [float("nan")] * len(es_texts)


# ========== Normalization and mixing ==========
def _minmax_series(values):
    """Min-max [0,1] per batch, ignoring NaN/inf; returns an aligned Series."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").astype(float).to_numpy()
    mask = np.isfinite(arr)
    out = np.full_like(arr, np.nan, dtype=float)
    if mask.sum() >= 2:
        v = arr[mask]
        mn, mx = np.min(v), np.max(v)
        if mx > mn:
            out[mask] = (arr[mask] - mn) / (mx - mn)
    return pd.Series(out)


# ========== Plots ==========
def plot_similarity(outdir, sid, sbert_cos, cosn, qe_mix):
    labels = []
    values = []
    if sbert_cos is not None and not (isinstance(sbert_cos, float) and np.isnan(sbert_cos)):
        labels.append("SBERT ES↔EN")
        values.append(max(0.0, min(1.0, float(sbert_cos))))
    if CONFIG.get("SURFACE_SIM_ENABLE", True) and cosn is not None and pd.notna(cosn):
        labels.append(f"cos{int(CONFIG['CHAR_N'])}(heur)")
        values.append(max(0.0, min(1.0, float(cosn))))
    if qe_mix is not None and not (isinstance(qe_mix, float) and np.isnan(qe_mix)):
        labels.append("QE_mix")
        values.append(max(0.0, min(1.0, float(qe_mix))))
    if not labels:
        return
    plt.figure()
    plt.bar(labels, values)
    plt.title(f"Similarity / Quality – sent_id {sid}")
    plt.ylim(0, 1)
    ensure_dir(outdir)
    plt.savefig(os.path.join(outdir, f"sent_{sid}_similarity.png"), bbox_inches="tight")
    plt.close()


def plot_structure(outdir, sid, length_ratio, digit_match, punc_jacc):
    length_dev = abs(1.0 - (length_ratio or 0.0))
    plt.figure()
    plt.bar(["|len_dev|", "digit_match", "punct_jacc"], [
        max(0.0, min(1.0, length_dev)),
        max(0.0, min(1.0, (digit_match or 0.0))),
        max(0.0, min(1.0, (punc_jacc or 0.0)))
    ])
    plt.title(f"Structure/Consistency – sent_id {sid}")
    plt.ylim(0, 1)
    ensure_dir(outdir)
    plt.savefig(os.path.join(outdir, f"sent_{sid}_structure.png"), bbox_inches="tight")
    plt.close()


# ========== HTML ==========
def write_html_report(path, df, stem, weights_note: str = ""):
    def fmt(x):
        return f"{x:.3f}" if (isinstance(x, (float, int, np.floating)) and pd.notna(x)) else "NA"

    rows = []
    for _, r in df.iterrows():
        sid = str(r["sent_id"]).strip()
        rows.append(f"""
        <tr>
          <td><b>{sid}</b></td>
          <td><b>ES:</b> {r['es_text']}<br><b>EN:</b> {r['en_text']}</td>
          <td>
            <div>COMET-QE: {fmt(r.get('comet_qe'))} (norm: {fmt(r.get('comet_qe_norm'))})</div>
            <div>SBERT ES↔EN: {fmt(r.get('sbert_cosine'))} (norm: {fmt(r.get('sbert_es_en_norm'))})</div>
            {"<div>cosN (heur): " + fmt(r.get('cosine_charN_text')) + f" (N={int(r.get('charN', 0))})</div>" if CONFIG.get('SURFACE_SIM_ENABLE', True) else ""}
            <div>QE_mix: {fmt(r.get('qe_mix'))}</div>
            <div>len_ratio: {fmt(r['length_ratio_en_over_es'])} | digits: {int(r['digit_match_binary'])} | punct: {fmt(r['punctuation_jaccard'])}</div>
            <div>flags: len={int(r['flag_len'])}, digits={int(r['flag_digits'])}{(", sbert_low="+str(int(r['flag_sbert_low']))) if 'flag_sbert_low' in r else ""}{(", cosN_low="+str(int(r['flag_cosine_low']))) if 'flag_cosine_low' in r else ""}{(", qe_mix_low="+str(int(r['flag_qe_mix_low']))) if 'flag_qe_mix_low' in r else ""}</div>
          </td>
          <td><img src="plots/sent_{sid}_similarity.png" style="max-width:340px"></td>
          <td><img src="plots/sent_{sid}_structure.png" style="max-width:340px"></td>
        </tr>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>ES–EN Parallel Corpus Report – {stem}</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
th {{ background: #f4f4f4; text-align: left; }}
tr:nth-child(even) {{ background: #fafafa; }}
.note {{ margin: 0.5rem 0; color: #444; }}
</style></head>
<body>
<h1>ES–EN Parallel Corpus Report – {stem}</h1>
<p>Sorted by <code>sent_id</code> (ascending). {weights_note}</p>
<table>
  <thead><tr>
    <th>sent_id</th><th>Texts</th><th>Metrics</th><th>Similarity Plot</th><th>Structure Plot</th>
  </tr></thead>
  <tbody>
    {''.join(rows)}
  </tbody>
</table>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ========== IDs helper ==========
def expand_ids_arg(ids_arg: Optional[str]) -> Optional[List[str]]:
    if not ids_arg:
        return None
    items = []
    for chunk in ids_arg.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            if a.isdigit() and b.isdigit():
                a, b = int(a), int(b)
                rng = range(a, b + 1) if a <= b else range(b, a + 1)
                items.extend([str(i) for i in rng])
            else:
                items.append(part)
        else:
            items.append(part)
    return items or None


# ========== Pair discovery (enhanced for variants) ==========
def _normalize_stem(filename: str) -> Optional[str]:
    # Remove known suffixes like _es.conllu, _en.conllu, .conllu, .conll
    bn = os.path.basename(filename)
    m = re.match(r"(?i)(.+?)_(?:es|en)\.(?:conllu|conll)$", bn)
    if m:
        return m.group(1)
    m2 = re.match(r"(?i)(.+?)\.(?:conllu|conll)$", bn)
    if m2:
        return m2.group(1)
    return None


def discover_pairs(src_dir: str) -> List[Tuple[str, str, str]]:
    """
    Discover pairs for processing.
    Behavior:
      1) Normal pairing for *_es.* + *_en.* files (recursive).
      2) Variant grouping: for files sharing a prefix like `01_20`,
         if a `*_es.*` exists and there are other files with same prefix
         (e.g., GPT1/GPT2), create pairs: (f"{prefix}__{variant_tag}", es_path, variant_path).
    Returns a list of (stem, es_path, en_or_variant_path).
    """
    src_path = Path(src_dir).expanduser().resolve()
    if not src_path.exists():
        print(f"[ERR] Source directory does not exist: {src_path}")
        return []

    # search recursively for candidate files (accept .conllu and .conll)
    es_patterns = ["**/*_es.conllu", "**/*_es.conll", "**/*-es.conllu", "**/*-es.conll"]
    en_patterns = ["**/*_en.conllu", "**/*_en.conll", "**/*-en.conllu", "**/*-en.conll"]

    es_files = []
    en_files = []
    for pat in es_patterns:
        es_files.extend([str(p) for p in src_path.glob(pat)])
    for pat in en_patterns:
        en_files.extend([str(p) for p in src_path.glob(pat)])

    es_files = sorted(set(es_files))
    en_files = sorted(set(en_files))

    index = {}
    for p in es_files:
        stem = _normalize_stem(p) or os.path.splitext(os.path.basename(p))[0]
        index.setdefault(stem, {"es": None, "en": None})["es"] = p
    for p in en_files:
        stem = _normalize_stem(p) or os.path.splitext(os.path.basename(p))[0]
        index.setdefault(stem, {"es": None, "en": None})["en"] = p

    pairs = []
    for stem, d in index.items():
        if d.get("es") and d.get("en"):
            pairs.append((stem, d["es"], d["en"]))

    # --- Variant grouping by prefix (e.g., 01_20) ---
    all_conllu = sorted([str(p) for p in src_path.rglob("*.conllu")] + [str(p) for p in src_path.rglob("*.conll")])
    groups = {}
    for p in all_conllu:
        bn = os.path.basename(p)
        m = re.match(r"^(\d+_\d+)", bn)   # e.g., 01_20...
        if m:
            prefix = m.group(1)
        else:
            # fallback: use the part before the first '_' if there is one
            prefix = bn.split("_", 1)[0] if "_" in bn else os.path.splitext(bn)[0]
        groups.setdefault(prefix, []).append(p)

    variant_pairs = []
    for prefix, files in groups.items():
        # find the es file in the group
        es_file = None
        for f in files:
            if re.search(r"(?i)(?:_es|[-\.]es)\.(conllu|conll)$", os.path.basename(f)):
                es_file = f
                break
        if not es_file:
            continue
        # create pair for each other file in group
        for f in files:
            if os.path.normpath(f) == os.path.normpath(es_file):
                continue
            variant_tag = os.path.splitext(os.path.basename(f))[0]
            pseudo_stem = f"{prefix}__{variant_tag}"
            variant_pairs.append((pseudo_stem, es_file, f))

    # Merge variant pairs but avoid duplicates (by path)
    canonical_pairs_set = set((os.path.normpath(a), os.path.normpath(b)) for (_, a, b) in pairs)
    for stem, es_p, hyp_p in variant_pairs:
        key = (os.path.normpath(es_p), os.path.normpath(hyp_p))
        if key not in canonical_pairs_set:
            pairs.append((stem, es_p, hyp_p))

    # Diagnostics
    if not pairs:
        print(f"[DIAG] discover_pairs: found ES files: {len(es_files)}, EN files: {len(en_files)} under {src_path}")
        if es_files:
            print("[DIAG] sample ES files:\n  " + "\n  ".join(es_files[:5]))
        if en_files:
            print("[DIAG] sample EN files:\n  " + "\n  ".join(en_files[:5]))
        sample_groups = {k: v[:3] for k, v in list(groups.items())[:6]}
        if sample_groups:
            print("[DIAG] sample groups (prefix -> files):")
            for k, v in sample_groups.items():
                print(f"  {k}:")
                for fp in v:
                    print(f"    {fp}")
        if not es_files and not en_files and not groups:
            print("[DIAG] No .conllu/.conll files found. Check filenames or the COMMON_SOURCE_DIR value.")

    return sorted(pairs, key=lambda x: x[0])


# ========== Processing a pair ==========
def process_pair(stem: str, es_path: str, en_path: str, out_root: str, ids_arg: Optional[str]):
    outdir = ensure_dir(os.path.join(out_root, f"parallel_outputs_{stem}_ES_EN"))
    plot_dir = ensure_dir(os.path.join(outdir, "plots"))

    es = parse_conllu(es_path)
    en = parse_conllu(en_path)

    # Ordered intersection
    common_ids = sorted(set(es.keys()) & set(en.keys()), key=lambda x: (int(x) if str(x).isdigit() else x))
    wl = expand_ids_arg(ids_arg) if ids_arg else None
    if wl is not None:
        wl_set = set(wl)
        common_ids = [sid for sid in common_ids if sid in wl_set]

    es_texts = [es[sid]["text"] for sid in common_ids]
    en_texts = [en[sid]["text"] for sid in common_ids]

    # --- S-BERT ES↔EN (adequacy) ---
    sbert_sims = sbert_batch_cosine(es_texts, en_texts) if common_ids else []

    # --- COMET-QE (reference-less quality) ---
    comet_scores = comet_qe_score_batch(
        es_texts, en_texts,
        device=CONFIG.get("COMET_QE_DEVICE", "cpu"),
        batch_size=CONFIG.get("COMET_QE_BATCH_SIZE", 8)
    ) if len(es_texts) > 0 else []

    records = []
    for idx, sid in enumerate(common_ids):
        es_text, en_text = es[sid]["text"], en[sid]["text"]
        es_tok = [r for r in es[sid]["tokens"] if '-' not in r[0] and '.' not in r[0]]
        en_tok = [r for r in en[sid]["tokens"] if '-' not in r[0] and '.' not in r[0]]
        es_len, en_len = len(es_tok), len(en_tok)
        length_ratio = en_len / es_len if es_len else 0.0

        # Structural signals
        digit_match = 1.0 if extract_numbers(es_text) == extract_numbers(en_text) else 0.0
        punc_jacc = jaccard(punctuation_set(es_text), punctuation_set(en_text))

        # Optional surface heuristic
        cosn = cosine_from_ngrams(es_text, en_text, CONFIG["CHAR_N"]) if CONFIG.get("SURFACE_SIM_ENABLE", True) else None

        sbert_cos = sbert_sims[idx] if sbert_sims else None
        comet_qe = comet_scores[idx] if comet_scores else float("nan")

        # Flags
        flag_len = not (CONFIG["LEN_LO"] <= (length_ratio if length_ratio > 0 else 0.0) <= CONFIG["LEN_HI"])
        flag_dig = (digit_match == 0.0)
        flag_sbert = False
        if sbert_cos is not None:
            flag_sbert = (sbert_cos < CONFIG["SBERT_LOW"])
        flag_cosn = False
        if CONFIG.get("SURFACE_SIM_ENABLE", True) and cosn is not None:
            flag_cosn = (cosn < CONFIG["COSINE_CHARN_LOW"])

        records.append({
            "sent_id": sid,
            "es_text": es_text,
            "en_text": en_text,
            "es_tokens": es_len,
            "en_tokens": en_len,
            "length_ratio_en_over_es": length_ratio,
            "digit_match_binary": digit_match,
            "punctuation_jaccard": punc_jacc,
            "sbert_cosine": sbert_cos if (sbert_cos is not None) else float("nan"),
            "sbert_model": CONFIG["SBERT_MODEL"] if _HAS_SBERT else "NA",
            "comet_qe": comet_qe,
            "comet_qe_model": (CONFIG["COMET_QE_MODEL"] if _HAS_COMET_QE else "NA"),
            "cosine_charN_text": (cosn if cosn is not None else float("nan")),
            "charN": CONFIG["CHAR_N"] if CONFIG.get("SURFACE_SIM_ENABLE", True) else 0,
            "flag_len": int(flag_len),
            "flag_digits": int(flag_dig),
            "flag_sbert_low": int(flag_sbert),
            "flag_cosine_low": int(flag_cosn),
        })

    df = pd.DataFrame.from_records(records)

    # Stable numeric ordering by sent_id
    df["__sid_num"] = pd.to_numeric(df["sent_id"].astype(str).str.strip(), errors="coerce")
    df = df.sort_values(["__sid_num", "sent_id"], ascending=[True, True], kind="mergesort").drop(columns="__sid_num")

    # ===== Normalizations and QE_mix =====
    if CONFIG.get("QE_MIX", True):
        df["comet_qe_norm"] = _minmax_series(df["comet_qe"])
        df["sbert_es_en_norm"] = _minmax_series(df["sbert_cosine"])
        w = CONFIG.get("QE_MIX_WEIGHTS", {"comet": 0.6, "sbert": 0.4})
        df["qe_mix"] = (w.get("comet", 0.6) * df["comet_qe_norm"] +
                        w.get("sbert", 0.4) * df["sbert_es_en_norm"])
        low_thr = float(CONFIG.get("QE_MIX_LOW", 0.35))
        df["flag_qe_mix_low"] = (df["qe_mix"] < low_thr).astype(int)
    else:
        df["qe_mix"] = np.nan
        df["flag_qe_mix_low"] = 0

    # ===== Per-sentence plots =====
    for _, r in df.iterrows():
        sid = str(r["sent_id"])
        plot_similarity(plot_dir, sid, r.get("sbert_cosine"), r.get("cosine_charN_text"), r.get("qe_mix"))
        plot_structure(plot_dir, sid, r.get("length_ratio_en_over_es"), r.get("digit_match_binary"), r.get("punctuation_jaccard"))

    # ===== CSVs =====
    df.to_csv(os.path.join(outdir, "parallel_es_en_metrics.csv"), index=False, encoding="utf-8")
    df.loc[:, ["sent_id", "es_text", "en_text"]].to_csv(os.path.join(outdir, "parallel_corpus_aligned.csv"), index=False, encoding="utf-8")

    review_mask = (
            (df["flag_len"] == 1) |
            (df["flag_digits"] == 1) |
            (df["flag_sbert_low"] == 1) |
            (df["flag_qe_mix_low"] == 1) |
            (df["flag_cosine_low"] == 1)
    )
    df.loc[review_mask].to_csv(os.path.join(outdir, "needs_review.csv"), index=False, encoding="utf-8")

    # ===== HTML =====
    weights_note = ""
    if CONFIG.get("QE_MIX", True):
        w = CONFIG.get("QE_MIX_WEIGHTS", {"comet": 0.6, "sbert": 0.4})
        weights_note = f"<span class='note'>QE_mix = {w.get('comet', 0.6):.2f}·COMET_QE_norm + {w.get('sbert', 0.4):.2f}·SBERT_norm (alert threshold = {float(CONFIG.get('QE_MIX_LOW', 0.35)):.2f})</span>"
    report_html = os.path.join(outdir, "parallel_report.html")
    write_html_report(report_html, df, stem, weights_note=weights_note)
    print(f"[OK] HTML report: {report_html}")

    # ===== Per-pair summary =====
    agg = df.agg({
        "qe_mix": "mean",
        "comet_qe": "mean",
        "sbert_cosine": "mean",
        "digit_match_binary": "mean",
        "punctuation_jaccard": "mean",
        "length_ratio_en_over_es": "mean",
        "cosine_charN_text": "mean"
    }).to_frame(name="mean").reset_index().rename(columns={"index": "metric"})
    agg.to_csv(os.path.join(outdir, "summary_means.csv"), index=False, encoding="utf-8")

    print(f"[OK] {stem}: {len(df)} pairs processed. Outputs in {outdir}")
    if _HAS_SBERT:
        print(f"[OK] SBERT: {CONFIG['SBERT_MODEL']} (device={CONFIG.get('SBERT_DEVICE', 'cpu')})")
    if _HAS_COMET_QE:
        print(f"[OK] COMET-QE: {CONFIG['COMET_QE_MODEL']}")


# ========== MAIN (batch) ==========
def main():
    src_dir = CONFIG["COMMON_SOURCE_DIR"]
    out_root = ensure_dir(CONFIG["COMMON_OUTPUTS_DIR"])
    pairs = discover_pairs(src_dir)
    if not pairs:
        print(f"[ERR] No *_es.conllu / *_en.conllu pairs or variants found in: {src_dir}")
        print("[HINT] Check filenames (e.g. 01_20_es.conll) or configure CONFIG['ES_PATH']/['EN_PATH'] to process a single pair.")
        return
    print(f"[OK] Detected pairs ({len(pairs)}): " + ", ".join(stem for stem, _, _ in pairs))
    for stem, es_path, en_path in pairs:
        try:
            process_pair(stem, es_path, en_path, out_root, CONFIG.get("IDS"))
        except Exception as e:
            print(f"[ERR] Processing failed for {stem}: {e}")
            traceback.print_exc()


# --- Patch: safe launcher + env info prints for debugging ---
def print_env_info():
    try:
        import platform
        import numpy as np
        print(f"[ENV] Python: {platform.python_version()} ({platform.platform()})")
        print(f"[ENV] NumPy: {np.__version__} -> {np.__file__}")
        try:
            import torch
            print(f"[ENV] torch: {getattr(torch, '__version__', 'unknown')}")
        except Exception as e:
            print(f"[ENV] torch: import failed: {e}")
        try:
            from sentence_transformers import SentenceTransformer
            print("[ENV] sentence-transformers: available")
        except Exception as e:
            print(f"[ENV] sentence-transformers: import failed: {e}")
    except Exception as e:
        print("[ENV] Could not print environment info:", e, file=sys.stderr)


def safe_main():
    try:
        print("[INFO] Starting ES-EN pipeline")
        print_env_info()
        try:
            cfg = globals().get("CONFIG", None)
            if cfg:
                print("[INFO] CONFIG keys:", ", ".join(cfg.keys()))
                print("[INFO] COMMON_SOURCE_DIR:", cfg.get("COMMON_SOURCE_DIR"))
                print("[INFO] COMMON_OUTPUTS_DIR:", cfg.get("COMMON_OUTPUTS_DIR"))
                if cfg.get("ES_PATH") or cfg.get("EN_PATH"):
                    print("[INFO] ES_PATH:", cfg.get("ES_PATH"))
                    print("[INFO] EN_PATH:", cfg.get("EN_PATH"))
        except Exception:
            pass
        main()
    except Exception:
        print("[ERROR] Pipeline crashed with exception:", file=sys.stderr)
        traceback.print_exc()


if __name__ == "__main__":
    safe_main()
