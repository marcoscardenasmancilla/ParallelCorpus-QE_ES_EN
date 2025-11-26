# ES-EN Parallel Corpus Quality Estimation (QE)
CPU-first, OOM-resilient pipeline that aligns Spanish–English CoNLL-U sentences, computes sentence-level adequacy (COMET-QE and sentence-BERT cosine), runs structural checks (length ratio, digits, punctuation), offers optional char-ngram heuristic, and outputs per-pair CSVs, PNGs, an HTML report and aggregated metrics.

# ES-EN Parallel Corpus Quality Estimation (QE) | Pipeline v1.2-safe
**ES↔EN GPT / S-BERT / COMET-QE Pipeline**  
Detailed developer README describing architecture, configuration, and internals.

---

## Table of contents
1. [About](#about)  
2. [Key features](#key-features)  
3. [Quick start](#quick-start)  
4. [Configuration (`CONFIG`) explained](#configuration-config-explained)  
5. [Pipeline structure & data flow](#pipeline-structure--data-flow)  
   - discovery & grouping  
   - parsing (CoNLL‑U)  
   - encoding & scoring (S‑BERT, COMET‑QE)  
   - surface heuristics (char‑N, punctuation, digits)  
   - normalization & QE_mix  
   - reporting (CSVs, plots, HTML, summary)  
6. [Functions (mapping to code)](#functions-mapping-to-code)  
7. [Outputs & file format details](#outputs--file-format-details)  
8. [Robustness, failover & performance considerations](#robustness-failover--performance-considerations)  
9. [Troubleshooting & diagnostics](#troubleshooting--diagnostics)  
10. [Integration & extension points](#integration--extension-points)  
11. [Licensing & contact](#licensing--contact)

---

# About
This script (`ES_EN_GPT_sbert_cometqe_pipeline.py`) is a CPU-first, OOM-resilient pipeline to align Spanish–English parallel text stored in CoNLL‑U files, compute sentence-level translation adequacy metrics (COMET‑QE-based translation quality and sentence‑BERT cosine similarity), run structural sanity checks (length ratio, digits, punctuation), optionally compute a lightweight char‑ngram similarity heuristic, and produce per-pair summaries: CSVs, PNG plots, an HTML report, and aggregated metrics.

It is designed for reproducible batch processing in constrained environments (e.g., laptop or shared compute without guaranteed GPU) with careful thread control and progressive fallbacks.

---

# Key features
- Discovers pairs from a shared directory using flexible filename patterns and grouping heuristics (handles `*_es.conllu` + `*_en.conllu` and grouped variants like `01_20_*`).
- COMET‑QE scoring (no reference required): `src` -> `mt` evaluation.
- Multilingual S‑BERT cosine similarity ES↔EN (chunked encoding to avoid memory spikes).
- Surface heuristics: length ratio, digit matching, punctuation Jaccard.
- Optional char‑n‑gram cosine heuristic (fast, language-agnostic similarity).
- `QE_mix`: min‑max normalized composite score combining COMET and S‑BERT.
- Outputs per-pair folder with `parallel_es_en_metrics.csv`, `parallel_corpus_aligned.csv`, `needs_review.csv`, `summary_means.csv`, `parallel_report.html`, and a `plots/` directory with per‑sentence PNGs.
- Defensive programming: set safe environment variables, thread limits, CPU-first defaults, COMET OOM retry to CPU & smaller batches.

---

# Quick start

1. Place Spanish and English CoNLL‑U files in one folder (e.g. `COMMON_SOURCE_DIR`).
   - Accepted filename patterns: `*_es.conllu`, `*_en.conllu`, `.conll` variants, and grouped prefixes like `01_20_es.conllu` alongside `01_20_gpt1.conllu`.
2. Edit the `CONFIG` dictionary at the top of the script to set paths and model/device options.
3. (Optional) Create and activate a virtualenv and install dependencies (`pandas`, `numpy`, `matplotlib`, `sentence-transformers`, `unbabel-comet`, `torch`, etc.).
4. Run:
```bash
python ES_EN_GPT_sbert_cometqe_pipeline.py
```

---

# Configuration (`CONFIG`) explained

The `CONFIG` dict is the single user-editable block controlling pipeline behaviour.

Key keys (with defaults used in script):

- `COMMON_SOURCE_DIR` (str)  
  Directory containing all `.conllu/.conll` files to be discovered.

- `COMMON_OUTPUTS_DIR` (str)  
  Root directory where `parallel_outputs_{stem}_ES_EN/` folders are created.

- `IDS` (str|None)  
  Optional: restrict processing to specific `sent_id` values, accepts comma lists and ranges (e.g. `"1,3,5-8"`).

- Structural sanity thresholds:
  - `LEN_LO` (float) — lower bound for `length_ratio_en_over_es` (default 0.60)
  - `LEN_HI` (float) — upper bound for `length_ratio_en_over_es` (default 1.60)

- S‑BERT options:
  - `SBERT_ENABLE` (bool) — enable S‑BERT scoring
  - `SBERT_MODEL` (str) — HF model id (default: `paraphrase-multilingual-MiniLM-L12-v2`)
  - `SBERT_DEVICE` (str) — `"cpu"` or `"cuda"`
  - `SBERT_BATCH_SIZE` (int) — chunk size for encoding
  - `SBERT_LOW` (float) — threshold to flag low S‑BERT similarity

- COMET‑QE options:
  - `COMET_QE_ENABLE` (bool)
  - `COMET_QE_MODEL` (str) — e.g. `Unbabel/wmt20-comet-qe-da`
  - `COMET_QE_DEVICE` (str) — `"cpu"` recommended by default
  - `COMET_QE_BATCH_SIZE` (int) — conservative default (8)

- Composite QE:
  - `QE_MIX` (bool) — enable composite QE score
  - `QE_MIX_WEIGHTS` (dict) — e.g. `{"comet": 0.6, "sbert": 0.4}`
  - `QE_MIX_LOW` (float) — low threshold to flag QE_mix

- Surface heuristic options:
  - `SURFACE_SIM_ENABLE` (bool)
  - `CHAR_N` (int) — n for char-ngrams
  - `COSINE_CHARN_LOW` (float) — low threshold for char‑n similarity

---

# Pipeline structure & data flow

Below is a step-by-step description of what the script does and which functions implement each stage.

## 1. Safe environment setup
The script sets several environment variables **before** importing heavy libraries:
- `TOKENIZERS_PARALLELISM=false`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `KMP_DUPLICATE_LIB_OK=TRUE`

If PyTorch is importable and no CUDA is available, it calls `torch.set_num_threads(1)` to reduce CPU thread contention. This reduces memory pressure and race conditions on multi-threaded BLAS libraries.

_Code locations:_ top-of-file safe-mode env vars.

## 2. Discover pairs (`discover_pairs`)
- Scans `COMMON_SOURCE_DIR` recursively for `.conllu` and `.conll` files.
- Normalizes stems using `_normalize_stem` (removes `_es` / `_en` suffixes).
- Creates canonical pairs from matched `_es` + `_en` files.
- Additionally groups files by prefix (e.g., `01_20_*`) to support variants (GPT outputs, multiple hypotheses). For each group, it pairs the one ES file with each other file in the group as a "variant pair".
- Returns a sorted list of tuples: `(stem, es_path, en_or_variant_path)`.

Important: the grouping logic avoids duplicates and prints diagnostics if no pairs were found.

_Code:_ `discover_pairs`, `_normalize_stem`.

## 3. Parse CoNLL‑U files (`parse_conllu`)
- Reads CoNLL‑U formatted files and extracts per-sentence:
  - `sent_id` (from `# sent_id = ...`)
  - `text` (from `# text = ...`)
  - `tokens` (first 10 columns of token lines)
  - `lemmas` and `upos` for convenience
- The `parse_conllu` function flushes each sentence when encountering blank line separators.

Edge cases handled:
- Skips token lines with ranges (IDs containing '-' or '.').
- Graceful exception handling and error logs.

_Code:_ `parse_conllu`.

## 4. Prepare sentence lists and optional filtering
- Computes intersection of `sent_id` keys present in both ES and EN parsed dictionaries.
- Orders `sent_id` numerically when possible; supports string IDs.
- Applies `IDS` filter (if provided) using `expand_ids_arg` which supports ranges like `5-10`.

_Code:_ `expand_ids_arg`, sorting logic in `process_pair`.

## 5. Compute metrics
Metrics are computed per batch and then per-sentence:

### a) S‑BERT ES↔EN similarity (`sbert_batch_cosine`)
- If enabled and available, the function encodes ES and EN sentence lists in chunks (`SBERT_BATCH_SIZE`) to vectors (using `SentenceTransformer.encode` with `convert_to_tensor=True`).
- Vectors are normalized and cosine similarity computed via element-wise product and sum (fast in PyTorch tensors).
- Returns a list of per-sentence similarity floats (close to 1 = highly similar).

Notes:
- Encodings are done on configured `SBERT_DEVICE` (e.g., `"cpu"` or `"cuda"`).
- Chunking reduces peak memory usage and avoids OOMs.

_Code:_ `sbert_batch_cosine`.

### b) COMET‑QE (`comet_qe_score_batch`)
- If COMET is available, prepares `data = [{"src": es, "mt": en}, ...]` and calls the COMET model `predict()` method.
- Tries GPU inference if `COMET_QE_DEVICE` starts with `"cuda"` and torch reports CUDA available.
- On `RuntimeError` (typical OOM), retries on CPU with a smaller batch (e.g., 4).
- Returns list of per-sentence COMET scores or NaN on failure.
- Supports different COMET model checkpoints (downloaded via `download_model`).

_Code:_ `_gpu_available`, `comet_qe_score_batch`.

### c) Surface heuristics
For each sentence pair:
- `length_ratio_en_over_es` = `len(en_tokens) / len(es_tokens)` (0 if `es_len` is 0).
- `digit_match_binary`: 1.0 if the set of numeric strings in ES and EN match exactly; else 0.0.
- `punctuation_jaccard`: Jaccard similarity of punctuation characters.
- `cosine_charN_text`: cosine over char‑n‑grams (via `cosine_from_ngrams` using `char_ngrams`) if `SURFACE_SIM_ENABLE`.

_Code:_ `extract_numbers`, `punctuation_set`, `jaccard`, `char_ngrams`, `cosine_from_ngrams`.

## 6. Flagging & record assembly
For each sentence the script calculates boolean flags:
- `flag_len`: whether length ratio falls outside `[LEN_LO, LEN_HI]`.
- `flag_digits`: whether numbers differ.
- `flag_sbert_low`: S‑BERT similarity lower than `SBERT_LOW`.
- `flag_cosine_low`: char-N heuristic below `COSINE_CHARN_LOW`.

These are stored as integer 0/1 in the per-sentence record.

## 7. Normalization & QE_mix
If `QE_MIX` is enabled:
- `comet_qe_norm` and `sbert_es_en_norm` are computed using `_minmax_series` which:
  - converts to numeric, ignores NaN/inf, and min-max scales values within the current batch.
  - If insufficient finite values (<2), returns NaNs for normalized column.
- `qe_mix` = weighted sum of normalized COMET and normalized S‑BERT using `QE_MIX_WEIGHTS`.
- `flag_qe_mix_low` marks sentences below `QE_MIX_LOW`.

Note: normalizations are **batch-local**, not global across different pairs. This is intentional because absolute scales of COMET and S‑BERT are not comparable across corpora/batches.

## 8. Visualization
For each sentence:
- `plot_similarity` creates a compact bar plot containing available `SBERT ES↔EN`, `cosN(heur)`, and `QE_mix` values, saved as `plots/sent_{sent_id}_similarity.png`.
- `plot_structure` creates a bar plot with:
  - `|len_dev|` (absolute deviation from length ratio 1.0)
  - `digit_match` (0 or 1)
  - `punct_jacc` (0–1)
  saved as `plots/sent_{sent_id}_structure.png`.

These are lightweight Matplotlib figures saved in the per-pair `plots/` directory.

## 9. CSVs, HTML and summary
- `parallel_es_en_metrics.csv` — full per-sentence dataframe with metrics and flags.
- `parallel_corpus_aligned.csv` — minimal table with `sent_id`, `es_text`, `en_text`.
- `needs_review.csv` — subset where any of the important flags are triggered.
- `parallel_report.html` — human-friendly HTML report showing:
  - Sent_id, texts, metrics (raw and normalized), flags, and inline images of the two plots.
  - The template is simple HTML written by `write_html_report`.
- `summary_means.csv` — per-pair aggregated means for the main metrics.

---

# Functions (mapping to code)

Below is an index of important functions and what they do:

- `ensure_dir(p)` — create and return absolute path for output directories.
- `strip_diacritics(s)` — helper: removes accents for char‑n processing.
- `extract_numbers(s)` — regex-based numeric token extractor.
- `punctuation_set(s)` — set of punctuation chars in string.
- `jaccard(a,b)` — set Jaccard similarity.
- `char_ngrams(s, n)` — create normalized char‑n‑gram list.
- `cosine_from_ngrams(a,b,n)` — compute cosine similarity from n‑gram counts.
- `_batch_iter(seq, size)` — yields successive chunks.
- `parse_conllu(path)` — parse CoNLL‑U into dict keyed by `sent_id`.
- `sbert_batch_cosine(es_texts, en_texts)` — chunked S‑BERT encoding + cosine.
- `_gpu_available(device_str)` — checks torch.cuda availability.
- `comet_qe_score_batch(es_texts, en_texts, device, batch_size)` — robust COMET‑QE predictor with OOM handling.
- `_minmax_series(values)` — min–max normalization ignoring NaN/inf.
- `plot_similarity(outdir, sid, sbert_cos, cosn, qe_mix)` — writes similarity bar plot PNG.
- `plot_structure(outdir, sid, length_ratio, digit_match, punc_jacc)` — writes structure bar plot PNG.
- `write_html_report(path, df, stem, weights_note)` — creates the HTML overview for a pair.
- `expand_ids_arg(ids_arg)` — parses `IDS` syntax like `"1,3,5-8"`.
- `_normalize_stem(filename)` — normalize file stem names to match ES/EN pairs.
- `discover_pairs(src_dir)` — find canonical pairs and grouped variant pairs.
- `process_pair(stem, es_path, en_path, out_root, ids_arg)` — main per-pair processing function.
- `main()` / `safe_main()` — batch-loop and safe launcher with environment diagnostics.
- `print_env_info()` — prints Python/NumPy/torch/sentence-transformers availability.

---

# Outputs & file format details

Each output folder: `parallel_outputs_{stem}_ES_EN/`

**parallel_es_en_metrics.csv** (columns)
- `sent_id`, `es_text`, `en_text`  
- `es_tokens`, `en_tokens` (token counts used for length ratio)  
- `length_ratio_en_over_es`  
- `digit_match_binary` (1.0/0.0)  
- `punctuation_jaccard` (0–1)  
- `sbert_cosine` (raw SBERT cosine)  
- `sbert_model` (model id or `"NA"`)  
- `comet_qe` (raw COMET score or `NaN`)  
- `comet_qe_model`  
- `cosine_charN_text` (char-N heuristic or `NaN`)  
- `charN`  
- `flag_len`, `flag_digits`, `flag_sbert_low`, `flag_cosine_low`, `flag_qe_mix_low` (0/1 flags)  
- `comet_qe_norm`, `sbert_es_en_norm`, `qe_mix` (if `QE_MIX` enabled)

**parallel_corpus_aligned.csv**
- `sent_id`, `es_text`, `en_text`

**needs_review.csv**
- Subset of metrics where `flag_*` is 1. Use for manual inspection.

**parallel_report.html**
- Self-contained HTML referencing images in `plots/`. Provides a quick visual pass-through.

**plots/**
- `sent_{sent_id}_similarity.png`
- `sent_{sent_id}_structure.png`

**summary_means.csv**
- Rows: `qe_mix`, `comet_qe`, `sbert_cosine`, `digit_match_binary`, `punctuation_jaccard`, `length_ratio_en_over_es`, `cosine_charN_text` with `mean` values.

---

# Robustness, failover & performance considerations

- **CPU-first default**: COMET and S‑BERT default to CPU in the shipped `CONFIG` to avoid GPU OOMs and to be reproducible across machines.
- **Thread controls**: Environment variables limit parallelism in tokenizers and BLAS libs to prevent oversubscription.
- **Chunking**: S‑BERT encoding is performed in chunks (`SBERT_BATCH_SIZE`) to limit memory peaks.
- **COMET OOM handling**: on `RuntimeError` typical of OOM, the code retries prediction on CPU and with smaller batch size (defensive).
- **Batch-local normalization**: `_minmax_series` normalizes scores within the current pair/batch; this avoids mixing distributions across corpora, but makes `qe_mix` relative to the processed batch.
- **Diagnostics**: `print_env_info()` shows Python, NumPy, torch, and sentence-transformers availability, useful when opening issues.

---

# Troubleshooting & diagnostics

Common issues and fixes:

- **No pairs found**
  - Ensure `COMMON_SOURCE_DIR` points to the correct folder.
  - Confirm file names include `_es` / `_en` or are in grouped prefix style.
  - Run `discover_pairs` diagnostics: the script prints sample files & groups.

- **S‑BERT import errors**
  - Install `sentence-transformers` and compatible `transformers`/`torch`.
  - If not available, set `SBERT_ENABLE` to `False` and the pipeline will continue without S‑BERT columns.

- **COMET errors / OOMs**
  - Use CPU mode (`COMET_QE_DEVICE="cpu"`) or reduce `COMET_QE_BATCH_SIZE`.
  - Ensure internet access for model download on first run.
  - If model load fails due to license/permission, check Hugging Face model card.

- **Plots not generated**
  - Confirm Matplotlib installed and backend set to `Agg` (the script sets it).
  - Check write permissions in `COMMON_OUTPUTS_DIR`.

- **Strange normalizations (NaN in norm cols)**
  - Occurs when there are <2 finite values for a particular metric in the batch (e.g., all COMET scores are NaN). The pipeline leaves norm columns as NaN in that case.

Diagnostic helpers:
- `print_env_info()` — prints versions and module availability.
- The script prints `[DIAG]` messages if pair discovery finds anomalies.

---

# Integration & extension points

You can extend or adapt the pipeline at multiple places:

- Replace or add models:
  - Swap `SBERT_MODEL` to a larger multilingual model for better semantics.
  - Use alternative QE models (e.g., newer COMET checkpoints).

- Add reference-based metrics:
  - If reference translations become available, add BLEU/chrF or reference-based COMET.

- Add more heuristics:
  - Named‑entity matching, numeric tolerance (e.g., allow formatting differences), token-level alignment coverage, language detection checks.

- Parallelization:
  - Process pairs in parallel (e.g., with `concurrent.futures.ProcessPoolExecutor`) at the pair level if hardware allows. Keep per-process thread limits to avoid oversubscription.

- Output formats:
  - Write a JSONL output for downstream ML pipelines.
  - Add dashboards (e.g., streamlit) for interactive inspection.

---

# Licensing & contact
- GNU Affero General Public License v3.0.
- Author / maintainer: Dr. Marcos H. Cárdenas-Mancilla.
- Date of creation: November 17, 2025.
- For reproducibility issues, include the output of `print_env_info()` when opening issues.

---

## Final notes
This README documents design decisions and implementation details to help new developers and researchers understand and adapt the pipeline. The code is intentionally conservative and defensive to make it robust for research usage on modest hardware.

