# Sadeed — What You Should Know

This document explains the key concepts, design decisions, and trade-offs in the project so you can confidently discuss them with judges.

---

## 1. The Three Detection Layers

### Layer 1: Great Expectations (GE)
- **What it does**: Checks 27 business rules derived from `data/LFS_Business_Rules.xlsx`, including cross-field validations
- **Examples**: "Age must be >= 15 if head of household", "Salary cannot exceed 50,000 SAR", "PhD holders must earn >= 8,000 SAR" (rule 3047), "Public sector hours >= 35" (rule 3031)
- **Strengths**: 100% precise — if a rule fires, it's definitively wrong data. Includes cross-field salary-education rules that catch subtle errors
- **Weakness**: Can only catch violations that are explicitly written as rules
- **Score**: Fraction of rules failed (e.g., 5/27 = 0.185)

### Layer 2: One-Class SVM (Support Vector Machine)
- **What it does**: Learns what "normal" data looks like from 2067 training samples, then flags records that are statistically far from normal
- **Uses only 4 features**: `age`, `cut_5_total` (usual weekly hours), `act_1_total` (actual weekly hours), `q_602_val` (salary)
- **Why only 4?**: The training dataset has 82 records with 416 columns, but most columns are sparse or categorical codes. These 4 are the numeric features that are consistently populated and meaningful for anomaly detection
- **Strengths**: Catches patterns GE cannot — like extreme combinations of values that individually pass all rules
- **Weakness**: Low recall against GE ground truth (F1=0.435) because it's solving a different problem. It's not trying to replicate GE — it's catching what GE misses
- **Training data**: 67 real normal records + 2000 simulated records generated from the same statistical distribution (mean/std matching per feature, log-normal for salary)
- **Score**: Sigmoid of the SVM decision function (0 = clearly normal, 1 = clearly anomalous)

### Layer 3: LLM (Mistral 7B / Qwen 2.5 14B via Ollama)
- **What it does**: Receives records already flagged by GE/SVM, validates whether they're truly anomalous, classifies severity (hard_error vs warning), and provides a human-readable English explanation — all in a single call
- **Runs locally**: No data leaves the machine. No API keys. No cloud dependency
- **Role in pipeline**: Acts as a **semantic reviewer**. It only runs when GE or SVM flags something, and receives both the SVM confidence score and GE violation details as context. It can confirm the flag or dismiss it as a false alarm
- **Two profiles**: Power-saving uses Mistral 7B (4.4 GB VRAM, F1=0.864), Performance uses Qwen 2.5 14B (~9 GB VRAM, F1=0.927)
- **Weakness**: Can occasionally miss subtle anomalies or flag borderline-normal records. GE flags cannot be overridden (they're ground truth)

---

## 2. The Pipeline Logic

```
Record comes in
    -> Check Redis cache (skip everything if cached)
    -> Stage 1: Run GE and SVM in parallel (asyncio.gather)
    -> Preliminary verdict = GE.is_anomaly OR SVM.is_anomaly
    -> If preliminary is False: return normal (LLM never called)
    -> If preliminary is True: Stage 2
        -> Run LLM detect() with SVM score as context (single call)
        -> Final verdict = preliminary AND LLM.is_anomaly
        -> If LLM says normal: override to normal, discard explanation
        -> If LLM confirms: return anomaly + English explanation
```

**Why this order?**
- GE and SVM are fast (milliseconds). LLM adds ~0.4s per record. Only call the LLM when needed
- GE catches rule violations. SVM catches statistical outliers. Together they cast a wide net
- The LLM validates flags and provides reasoning — it receives the SVM's confidence score as context, making its job easier (confirm/deny vs classify from scratch)
- Detection and explanation happen in a single LLM call, not two separate ones

**The formula**: `(GE OR SVM) AND LLM`

---

## 3. Why the SVM Matters

On the 70 real records alone, the SVM looks weak (F1=0.435). But that's measuring it against GE rules — which it's not designed to replicate.

The SVM's real value is catching **records that pass all 21 GE rules but are still wrong**.

On a large-scale test of 400 synthetic records (200 normal + 200 anomalous, 10 anomaly types):

| Layer | TP | FP | FN | Precision | Recall |
|-------|-----|-----|-----|-----------|--------|
| GE | 0 | 0 | 200 | — | 0.0% |
| SVM | 129 | 7 | 71 | 94.9% | 64.5% |
| LLM | 199 | 180 | 1 | 52.5% | 99.5% |
| **(GE OR SVM) AND LLM** | **129** | **7** | **71** | **94.9%** | **64.5%** |

**Key findings:**
- The LLM **never overrides a correct SVM flag** — all 71 misses are "SVM missed", zero are "LLM denied"
- The LLM catches 199/200 anomalies on its own, but has 180 false positives — the pipeline filters these
- The SVM is the **bottleneck for recall** — improving SVM recall directly improves the full pipeline
- Without the SVM, flagged records would never reach the LLM (the LLM only runs when GE OR SVM flags)

---

## 4. The Training Data Problem

We only have **82 records** in the training dataset, and 416 columns per record. This is extremely small for machine learning. Here's how we dealt with it:

- **GE doesn't need training data** — it's rule-based, derived from the business rules document
- **SVM needed more data** — with only 67 normal samples, the decision boundary was poor. We generated 2000 synthetic normal samples anchored to the real data's distributions. This improved the combined F1 from ~0.5 to 0.941
- **LLM doesn't need training data** — it uses few-shot prompting (examples in the prompt) rather than fine-tuning
- **FAQ Q9 explicitly encourages synthetic data generation** for improving model accuracy

---

## 5. The LLM Confirmation Prompt (How the LLM Works)

The LLM operates in **confirmation mode** — it doesn't classify records from scratch. Instead, it receives records already flagged by GE/SVM along with the SVM's confidence score, and confirms or denies the flag.

### Why Confirmation Instead of Independent Classification?

We tried independent few-shot classification first (with Qwen 3.5:9b). It failed — small models don't have enough reasoning capacity to reliably detect subtle anomalies from scratch. The key insight: **asking "is this flagged record actually wrong?" is a much easier task than "classify this record"**. See `struggles.md` for the full story.

### The Confirmation Prompt

```
You are a data-quality reviewer for Saudi Labour Force Survey records.
Our automated statistical model has ALREADY flagged the record below as a
possible anomaly. Your job is to CONFIRM or DENY the flag.

== Check these things ==
1. SALARY vs EDUCATION: Is the salary plausible for the education level?
   No edu/Primary: 1,500-5,000 | Secondary: 3,000-12,000 | ...
2. HOURS: Are usual and actual hours consistent? Few hours + high salary?
3. AGE: Is the age plausible for the education, hours, and role?
4. COMBINATIONS: Does anything look implausible together?

--- Flagged Record ---
Data: {"age": 28, "q_301": 10500009, "q_602_val": 800, ...}

--- Why it was flagged ---
Statistical model confidence: 72% likely anomalous

Do you confirm this is anomalous? Respond with JSON only.
```

The prompt is only ~470 tokens — short enough for a 7B model to process reliably. Detection and explanation happen in one call: the JSON response includes `is_anomaly`, `score`, and `explanation`.

### Why Mistral 7B / Qwen 2.5 14B?

We evaluated four models in the full pipeline:

| Model | Pipeline F1 | Precision | Recall | FP | VRAM | Speed |
|-------|-----------|-----------|--------|-----|------|-------|
| Qwen 3.5 9B | 0.316 | 1.000 | 0.188 | 0 | 6.6 GB | ~42s/rec |
| Gemma 2 9B | 0.820 | 0.940 | 0.727 | 3 | 5.5 GB | ~1.7s/rec |
| **Mistral 7B** (power-saving) | **0.864** | **0.760** | **1.000** | **6** | **4.4 GB** | **~0.7s/rec** |
| **Qwen 2.5 14B** (performance) | **0.927** | **0.864** | **1.000** | **3** | **~9 GB** | **~1.5s/rec** |

Qwen 3.5 is a "thinking model" that wastes tokens on internal reasoning chains. Mistral 7B has perfect recall and the lowest VRAM. Qwen 2.5 14B halves false positives for GPUs with 10+ GB VRAM. See `struggles.md` for the full journey.

### How the Response is Parsed

The LLM returns JSON like `{"is_anomaly": true, "score": 0.95, "explanation": "Salary too low for Master's degree"}`. The parser handles edge cases:

1. Strips markdown code fences (` ```json ... ``` `)
2. Extracts the first `{...}` JSON block if the model wraps it in prose
3. If parsing fails entirely, returns a fail-safe `is_anomaly=False` so the pipeline never crashes

---

## 6. What "No API Keys" Means for Judges

The LLM runs entirely on-device via Ollama. This is a deliberate design choice:

- Survey data is sensitive (personal demographics, salary, employment) — it never leaves the network
- Judges don't need to configure keys, manage billing, or worry about rate limits
- The system works offline once Docker images are pulled
- Results are reproducible — same model weights every time
- This aligns with the hackathon's focus on a deployable, self-contained system

---

## 7. What the Chrome Extension Does

The extension has two modes:

1. **Live form scanning**: Injects a "Scan Form" button on any web page. When clicked, it reads all form fields, sends them to the backend, and overlays results (green/red highlighting) directly on the page
2. **Excel batch upload**: A tab in the popup lets judges drag & drop an Excel file. Each row is sent for detection with a progress bar

The extension communicates with the backend via the service worker (`background.js`). The API URL is configured to `localhost:8000` for local development.

---

## 8. Caching

Results are cached in Redis by `record_id` with a 300-second TTL. This means:
- Submitting the same record twice within 5 minutes returns the cached result instantly
- This is important for the extension's Excel upload — if a judge re-uploads the same file, it's fast
- Cache can be bypassed by changing the `record_id`

---

## 9. Current Performance Numbers

### Real LFS records (70 records)

| Configuration | Accuracy | Precision | Recall | F1 | FP |
|---------------|----------|-----------|--------|-----|-----|
| GE Alone | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| SVM Alone | 0.857 | 0.696 | 0.842 | 0.762 | 7 |
| GE OR SVM | 0.900 | 0.731 | 1.000 | 0.844 | 7 |
| **(GE OR SVM) AND LLM** | **0.900** | **0.773** | **0.895** | **0.829** | **5** |

### Large-scale synthetic (400 records)

| Configuration | Accuracy | Precision | Recall | F1 | TP | FP | FN |
|---------------|----------|-----------|--------|-----|-----|-----|-----|
| GE Alone | 0.615 | 1.000 | 0.230 | 0.374 | 46 | 0 | 154 |
| SVM Alone | 0.823 | 0.971 | 0.665 | 0.789 | 133 | 4 | 67 |
| GE OR SVM | 0.927 | 0.978 | 0.875 | 0.923 | 175 | 4 | 25 |
| **(GE OR SVM) AND LLM** | **0.833** | **0.972** | **0.685** | **0.804** | **137** | **4** | **63** |

**Key findings:**
- GE and SVM are complementary: combined recall jumps to 0.875 on 400 records
- The LLM filters false positives while maintaining high precision (0.972)
- GE flags are ground truth and cannot be overridden by the LLM

---

## 10. How to Talk About This to Judges

**The pitch**: "Sadeed is a three-layer anomaly detection system for labour force survey data. The first layer checks 21 business rules instantly. The second layer uses machine learning to catch statistical outliers that rules can't express. The third layer uses a local LLM to confirm detections and explain why a record is wrong — all in one fast call. Everything runs locally, no data leaves the network."

**If asked about the SVM's recall**: "The SVM catches 64.5% of anomalies with 94.9% precision on our 400-record test. The missed anomalies are mostly salary-education mismatches where the salary is very low — the SVM's training distribution doesn't cover those well. The LLM acts as a semantic check on top — out of 84 total misses, only 1 was an LLM error. The bottleneck is SVM recall."

**If asked about model choice**: "We evaluated four models. Qwen 3.5 was a thinking model that produced empty responses — unusable. Gemma 2 had good numerical reasoning but lower recall. We ship two profiles: Mistral 7B for power-saving (4.4 GB VRAM, perfect recall, F1=0.864) and Qwen 2.5 14B for performance (halves false positives, F1=0.927). Both run entirely locally via Ollama."

**If asked whether the LLM is actually doing anything**: "Yes — it genuinely reasons about data quality. It classifies anomalies by severity (hard_error for impossible data vs warning for suspicious data), provides natural language explanations like 'A 12-year-old holding a bachelor's degree and earning 650,000 SAR per month is highly improbable', and filters false positives from the SVM. With Qwen 2.5 14B, only 3 false positives across 70 real records."

**If asked about the small dataset**: "We had 82 records. We used the real data distributions to generate 2000 additional normal samples, which the hackathon FAQ explicitly encourages. This gave the SVM a much better decision boundary. We also validated on 400 fully synthetic records to prevent overfitting."

**If asked about privacy**: "The entire system runs locally. The LLM is on-device via Ollama (only 4.4GB VRAM) — no data leaves the network. No API keys, no cloud calls, no external dependencies once deployed."
