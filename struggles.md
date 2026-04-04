# Sadeed — Struggles & Lessons Learned

## 1. Test fixture missing derived column (`edu_ordinal`)

**Problem:** `test_metrics.py` loaded 0 real records because the `real_records` fixture filtered for `edu_ordinal` in raw Excel data, but that column is derived from `q_301` inside `SVMDetector._preprocess`.

**Fix:** Excluded `edu_ordinal` from the raw-data completeness check since the SVM derives it on the fly.

**Lesson:** When testing a pipeline that has preprocessing stages, fixtures must reflect what exists *before* preprocessing, not after.

---

## 2. LLM returns `"score": null` — crashes `float(None)`

**Problem:** `parsed.get("score", 0.0)` returns `None` when the key exists but the value is JSON `null`. `float(None)` raises `TypeError`.

**Fix:** Changed to `float(parsed.get("score") or 0.0)` — the `or` coalesces both missing keys and null values.

**Lesson:** `dict.get(key, default)` only uses the default when the key is *absent*, not when the value is `None`.

---

## 3. qwen3.5 thinking model causes extreme generation times

**Problem:** Without `num_predict`, qwen3.5:9b generates very long `<think>` chains (internal reasoning). Some records took 2+ minutes. The `stream: False` + `timeout: 120s` combo didn't help because Ollama keeps the HTTP connection alive during generation — httpx's read timeout never fires.

**Fix:** Added `"options": {"num_predict": 2048}` to the Ollama payload to cap total output tokens.

**Lesson:** Thinking models (qwen3, deepseek-r1, etc.) need explicit token limits. HTTP timeouts don't work reliably with `stream: False` because the server is actively generating, not idle.

---

## 4. Longer prompts = worse performance on 9B model

**Problem:** We tried adding verbose reasoning explanations to few-shot examples to teach the model *why* records are normal/anomalous. Results got dramatically worse:

| Prompt style | Tokens | LLM Recall | Synth |
|---|---|---|---|
| Tighter rules, no reasoning | ~1,500 | 0.438 | 3/10 |
| Verbose reasoning (4 lines each) | ~2,250 | 0.000 | 0/10 |
| Concise reasoning (1 line each) | ~1,800 | 0.125 | 2/10 |
| Tightest (3N/7A, no reasoning) | ~1,370 | 0.688 | 3/10 |

**Lesson:** Small models (9B) have limited effective context. More prompt text doesn't help — it overwhelms the model. The model fixates on surface patterns (majority class) when the prompt is too long. Shorter, more direct prompts outperform verbose ones at this scale.

---

## 5. "When in doubt, classify as normal" killed recall

**Problem:** The original system instructions said "Only flag records with clear, significant violations — not borderline or unusual-but-plausible values. When in doubt, classify as normal." The LLM took this literally — recall was 31%, it missed obvious anomalies like a PhD earning 500 SAR.

**Fix:** Replaced with "It is better to flag a borderline record than to let a bad record through" and added explicit anomaly rules (salary below range is as bad as above, hours gap > 30, etc.).

**Lesson:** LLMs follow instructions literally. Conservative guardrails in classification prompts destroy recall. Be explicit about what "anomalous" means — don't rely on the model's judgment for domain-specific thresholds.

---

## 6. Few-shot class ratio matters

**Problem:** With 5 normal / 3 anomaly examples the model was biased toward "normal". Changing to 3 normal / 7 anomaly improved recall from 44% to 69%.

**Lesson:** Few-shot class balance directly influences the model's prior. For anomaly detection where recall matters, over-represent the anomaly class in examples.

---

## 7. Overfitting to synthetic test set

**Problem:** After tuning the prompt and few-shot examples, LLM scored 3/10 on the synthetic test set. But when we created a completely new validation set with distinct anomaly patterns, it dropped to 1/10. Several of the few-shot anomaly examples were near-copies of the test cases.

**Fix:** Created a fresh validation set with different ages, genders, education levels, and violation types. No overlap with few-shot examples.

**Lesson:** Always validate on held-out data. It's easy to accidentally tune the prompt to the test set, especially when few-shot examples resemble test cases.

---

## 8. Independent classification vs. confirmation mode

**Problem:** Asking the LLM to classify records from scratch (independent few-shot classification) was too hard for a 9B model. Even with optimized prompts, recall on unseen anomalies was poor.

**Approach (in progress):** Restructured the LLM's role from "classify this record" to "our SVM flagged this with 85% confidence — do you agree?" This is a much easier task:
- Shorter prompt (~470 tokens vs ~1,370)
- Simpler decision (confirm/deny vs classify from scratch)
- SVM score provides an anchor for the model's reasoning
- No few-shot examples needed

**Lesson:** Match the task difficulty to the model's capability. A 9B model can reason about *why* a flagged record is wrong, but it can't reliably detect subtle anomalies from scratch.

---

## 9. `num_predict` limit vs. thinking tokens

**Problem (emerging):** With `num_predict: 2048`, qwen3.5 sometimes spends all tokens on `<think>` blocks and produces an empty response (no JSON output). This triggers the fail-safe (`is_anomaly=false, score=0.0`), silently killing recall.

**Status:** Confirmed — the full test showed LLM recall of 18.8% with most misses being fail-safes from empty responses. Tried `think: false` in Ollama options and `/no_think` in prompt — neither worked reliably.

**Resolution:** Switched from qwen3.5:9b to **mistral** (7B, non-thinking model). See #10.

---

## 10. Switching from qwen3.5:9b to mistral — the breakthrough

**Problem:** qwen3.5:9b was fundamentally unsuited for the confirmation gate role:
- Thinking model wasted tokens on `<think>` chains, producing empty responses
- 6.6GB VRAM footprint
- 36 min per test run (~42s/record)
- Best recall on fresh validation set: 18.8% (1/10 synthetic)

**Fix:** Switched to `mistral` (7B, non-thinking, 4.4GB):

| Metric | qwen3.5:9b | mistral |
|---|---|---|
| Runtime | 36 min | **84 seconds** |
| LLM Recall | 0.188 | **1.000** |
| Synth caught | 1/10 | **10/10** |
| Full pipeline F1 | 0.316 | **0.970** |
| Full pipeline FP | 0 | **1** |
| VRAM | 6.6 GB | **4.4 GB** |

**Lesson:** Thinking models (qwen3, deepseek-r1) are not always better. For structured tasks like "confirm this JSON classification", a smaller non-thinking model that reliably outputs the requested format outperforms a larger thinking model that burns tokens on internal reasoning. Match the model architecture to the task shape.

---

## 11. Arabic explanations — abandoned in favour of English

**Problem:** Mistral's Arabic output was grammatically rough and sometimes focused on the wrong field. A 7B model just doesn't have strong enough Arabic language capabilities to produce professional-quality explanations.

**Fix:** Switched to English-only explanations. The `detect()` call already returns an `explanation` field in its JSON response — clear, accurate English reasoning like "Salary 800 SAR too low for Master's degree (expected 8,000-35,000)". Dropped the separate `explain()` call entirely.

**Bonus:** This also halved the LLM latency per record (one call instead of two), bringing the full test from 84s to 36s.

**Lesson:** Don't force a model to do something it's not good at. English explanations from a 7B model are clear and useful. Arabic from the same model was confusing and sometimes wrong — worse than no explanation at all.

---

## 12. Validating at scale — 400 synthetic records

**Problem:** Early tuning used the same 10 synthetic anomalies as the test set. Results looked good but we were overfitting — when we swapped in 10 fresh anomalies, the LLM dropped from 3/10 to 1/10.

**Fix:** Built a large-scale test generator (`test_metrics_large.py`) producing 400 records: 200 normal + 200 anomalous across 10 different anomaly types (salary too high/low, hours mismatches, age-education conflicts, compound anomalies). Seeded for reproducibility.

**Results:** On 400 records in 3 minutes:
- SVM: 94.9% precision, 64.5% recall (129/200 caught)
- LLM: 99.5% recall (199/200) but 180 false positives alone
- Full pipeline: 94.9% precision, 64.5% recall — LLM never overrides a correct SVM flag
- All 71 misses are "SVM missed", zero "LLM denied"

**Lesson:** Always validate on data that's larger than and separate from your tuning set. The 10-sample test gave a misleading picture. The 400-sample test revealed the true bottleneck (SVM recall, not LLM accuracy).

---

## 13. Mistral misreads small numbers — switched to Gemma 2

**Problem:** Mistral 7B consistently misread small salary values. `q_602_val: 24` was interpreted as "2400 SAR" or "24,000 SAR". Even adding explicit notes ("q_602_val=24 means exactly 24 SAR") didn't help — the model hallucinated multipliers. This isn't one edge case; any small or ambiguous number could be misread.

**Fix:** Switched to **Gemma 2 9B** — same speed (~0.4s/record), slightly more VRAM (5.5GB vs 4.4GB, still under 7GB), but much stronger numerical reasoning. On the same test: correctly reads 24 as 24 SAR, 800 as 800 SAR, etc.

**Lesson:** Not all 7B-class models are equal at numerical reasoning. Gemma 2 handles structured data with small numbers far better than Mistral. Test with edge-case values (very small, very large) when choosing a model for data validation tasks.

---

## 14. LLM was a yes-machine — confirmation bias in the prompt

**Problem:** In confirmation mode, the LLM flagged 100% of records as anomalous — including all 200 normal records in the 400-record test. The prompt said "Our model has ALREADY flagged this" and "if you find ANY problem, respond true", which anchored the LLM to always agree.

The pipeline metrics were identical with or without the LLM (same F1, same FP). A judge looking at the numbers would see the LLM adds zero detection value.

**Fix:** Rewrote the confirmation prompt to be neutral:
- Removed "ALREADY flagged" anchoring — replaced with "statistical models often produce FALSE POSITIVES"
- Added explicit examples of normal records that should NOT be flagged
- Changed from "any problem → true" to specific thresholds
- Added "do NOT flag records just because the statistical model did"

**Results (400 records):**

| | Before (yes-machine) | After (balanced) |
|---|---|---|
| LLM FP | 200 | **67** (-66%) |
| LLM Precision | 50.0% | **70.7%** |
| LLM Recall | 100% | **81.0%** |
| LLM F1 | 0.667 | **0.755** |

The LLM now demonstrably reasons — it rejects most false alarms while catching real anomalies. Only 1 "LLM denied" out of 84 missed anomalies; the rest are SVM misses.

**Lesson:** Confirmation prompts create anchoring bias. If you tell a model "this was flagged, confirm?", it will almost always confirm. Frame the task neutrally: "this might be a false alarm — check it."

---

## 15. GE explanations were unreadable — enriched with human-readable labels

**Problem:** GE's explanation for the 12-year-old record was: "age: rule expect_column_pair_values_A_to_be_greater_than_B failed" repeated 4 times. Completely useless.

**Fix:** 
1. Added a `_RULE_LABELS` dict mapping internal rule column names to human-readable descriptions (e.g. `_rule_2013_min_bachelor_age` → "Bachelor's requires age >= 21")
2. Enriched GE output with `failed_details` including column names, actual values, and expected ranges
3. For GE-flagged records, the LLM receives these findings as context so it can write comprehensive explanations covering ALL violations

**Before:** "age: rule expect_column_pair_values_A_to_be_greater_than_B failed"
**After:** "Bachelor's requires age >= 21; Under 15 must be never married; q_602_val value 650000 is outside allowed range (500-50000)"

And the LLM top-level explanation: "A 12-year-old holding a bachelor's degree and earning 650,000 SAR per month is highly improbable."

**Lesson:** Internal system names should never reach the user. Map every machine-readable identifier to a human-readable label at the boundary.

---

## 16. "Too good to be true" — original metrics hid the real problem

**Problem:** The first metrics test (70 real records + 10 synthetic) showed F1=0.970, Precision=0.941, Recall=1.000. Looked amazing. But these metrics were misleading:
- The 70 real records used GE as ground truth — so GE always scored perfectly against itself
- The 10 synthetic anomalies were hand-crafted to match the LLM's few-shot examples
- SVM was only tested on data similar to its training distribution

**Fix:** Created a 300-sample test (`test_metrics_300.py`) with 150 normal + 150 anomalous records across 10 anomaly types, all generated from a seeded RNG. The real picture emerged: F1=0.744, Recall=0.600. Four out of ten anomalies were slipping through.

**Lesson:** Always stress-test with large, diverse, independently generated data. Small hand-crafted test sets create a false sense of security.

---

## 17. SVM threshold tuning — 0.2 was the sweet spot (then it wasn't)

**Problem:** The SVM's binary decision (score > 0.5) was the gate for LLM calls. But many true anomalies had SVM scores between 0.2-0.5 — the SVM sensed something was off but not enough to flag. These never reached the LLM.

**Fix 1:** Added a configurable `svm_llm_threshold` parameter. Lowering it to 0.2 sent borderline records to the LLM, boosting F1 from 0.744 to 0.805 (+0.061) with only 7 more false positives.

**Plot twist:** After adding GE cross-field rules (struggle #19), the 0.2 threshold became unnecessary — GE caught the salary-education cases directly. The optimal threshold reverted to 0.5. The borderline path still exists in the code for deployments that need it.

**Lesson:** Optimizations interact. A tuning that helps in one configuration may become irrelevant (or harmful) after a different improvement. Re-evaluate all parameters when the system changes.

---

## 18. Ensemble LLM — a good idea that backfired

**Problem:** With ~14GB VRAM free, we tried ensembling Gemma 2 9B + Mistral 7B (both must agree to flag an anomaly). The theory: Gemma's precision + Mistral's recall = fewer false positives.

**Results on 300 records:**

| Config | Prec | Rec | F1 | FP |
|---|---|---|---|---|
| Pipeline + Gemma | 0.940 | 0.727 | 0.820 | 7 |
| Pipeline + Mistral | 0.917 | 0.807 | 0.858 | 11 |
| Pipeline + Ensemble (AND) | 0.933 | 0.653 | 0.769 | 7 |

The ensemble killed recall (-15%) while only eliminating 1 false positive. Mistral flags 89 records Gemma doesn't — many real anomalies. The AND gate threw all of those away.

Also: Ollama context-switches between models in VRAM, causing 14s/record (vs 1.5s for a single model). The 45-minute test run was painful.

**Surprise finding:** Mistral alone (F1=0.858) beat Gemma alone (F1=0.820) in the pipeline, despite Gemma having better numerical reasoning in isolation. Mistral's aggression is an *advantage* when the SVM gate pre-filters most records.

**Lesson:** Model ensembles aren't free. The AND gate between two imperfect classifiers doesn't improve results when their error patterns overlap on different subsets. The "aggressive model behind a conservative gate" pattern beat the "two moderate models must agree" pattern.

---

## 19. GE cross-field rules — the biggest single improvement

**Problem:** GE only validated ranges and null checks. Salary was valid if 500-50000 SAR regardless of education. A PhD earning 800 SAR passed GE, scored 0.3 on SVM (below threshold), and never reached the LLM. The preprocessing already computed `_rule_edu_salary_*` columns but they were **never wired into the suite**.

**Fix:** Added 8 new expectations to `suite.json`:
- Salary caps by education: No edu <= 5000, Secondary <= 15000, Diploma <= 25000
- Salary floors for high education: Bachelor's >= 3000, Master's >= 5000, PhD >= 8000
- Hours mismatch: |usual - actual| > 40

**Results:**

| Metric | Before | After |
|---|---|---|
| F1 | 0.744 | **0.871** (+0.127) |
| Precision | 0.978 | **0.975** |
| Recall | 0.600 | **0.787** (+0.187) |
| GE alone recall | 0.000 | **0.660** |

GE alone now catches 66% of anomalies — for free, with no LLM calls and near-perfect precision. The cross-field rules catch exactly the cases that both SVM and LLM struggled with (salary-education mismatches near boundaries).

**Also:** Since GE flags are ground truth (LLM can't override), these detections are rock-solid. The SVM threshold could revert to 0.5 because GE handles the borderline cases directly.

**Lesson:** Don't reach for ML when deterministic rules will do. The salary floor rules are obvious domain knowledge that should have been there from the start. A few lines of `if salary < 5000 and edu == PhD` outperformed weeks of prompt engineering and model selection.

---

## 20. Mistral's return — re-evaluation after GE rules

**Problem:** We originally switched from Mistral to Gemma 2 because Mistral misread small salary values (struggle #13). But with the new GE salary floor rules, those cases are now caught by GE before reaching the LLM.

**Re-evaluation:** Tested all three models again with the updated GE rules:

| Model | Pipeline F1 | VRAM | Speed |
|---|---|---|---|
| Gemma 2 9B | 0.820 | 5.5 GB | ~1.7s/rec |
| **Mistral 7B** | **0.858** | **4.4 GB** | **~1.5s/rec** |

Mistral's weakness (small number misreading) no longer matters because GE catches those cases deterministically. Its strength (higher recall on the records that do reach the LLM) makes it the better choice.

**Lesson:** Re-evaluate model selection after changing other parts of the pipeline. A model's weaknesses may become irrelevant if another layer compensates.
