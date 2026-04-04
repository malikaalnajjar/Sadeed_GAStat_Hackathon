"""
LLM-based few-shot anomaly detection via Ollama.

Constructs a few-shot prompt from module-level labelled examples, appends the
candidate record, and parses the model's JSON response to extract the anomaly
verdict, a confidence score, and a brief explanation.

The strategy is designed to be **fail-safe**: any network or parsing failure
returns ``is_anomaly=False`` with ``score=0.0`` and logs a warning so the
rest of the detection pipeline continues uninterrupted.

Dependencies:
    httpx  (async HTTP client for the Ollama REST API)

Ollama endpoint used:
    POST /api/generate
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from backend.detection.base import BaseDetector
from backend.models.schemas import StrategyName, StrategyResult

# ---------------------------------------------------------------------------
# Human-readable code mappings for LLM prompt enrichment
# ---------------------------------------------------------------------------

_CODE_LABELS: dict[str, dict[int, str]] = {
    "gender": {1600001: "Male", 1600002: "Female"},
    "family_relation": {
        1700001: "Head of household", 1700002: "Spouse (old code)",
        1700003: "Brother", 1700004: "Father", 1700005: "Mother",
        1700006: "Sister", 1700007: "Grandfather", 1700008: "Grandmother",
        1700009: "Grandchild", 1700010: "Domestic worker", 1700011: "Other relative",
        1700021: "Spouse", 1700022: "Son", 1700023: "Daughter", 1700030: "Other",
    },
    "marage_status": {
        10600001: "Never married", 10600002: "Married",
        10600003: "Divorced", 10600004: "Widowed", 10600012: "Unknown",
    },
    "nationality": {1800001: "Saudi"},
    "q_301": {
        10500031: "No formal education", 10500011: "Primary incomplete",
        10500003: "Primary", 10500017: "Intermediate",
        10500019: "Secondary", 10500020: "Secondary vocational",
        10500021: "Associate diploma", 10500023: "Diploma",
        10500025: "Bachelor's", 10500026: "Bachelor's (5yr)",
        10500009: "Master's", 10500010: "PhD",
    },
    "q_534": {
        99400001: "Public sector", 99400002: "Semi-public",
        99400003: "Private sector", 99400004: "Domestic worker",
    },
}


def _humanise_record(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *data* with coded fields decoded to readable labels.

    E.g. ``{"q_301": 10500010}`` → ``{"education": "PhD (10500010)"}``
    """
    _FIELD_RENAMES = {
        "q_301": "education",
        "q_602_val": "monthly_salary_SAR",
        "cut_5_total": "usual_weekly_hours",
        "act_1_total": "actual_weekly_hours",
        "marage_status": "marital_status",
        "q_534": "job_sector",
    }
    out: dict[str, Any] = {}
    for key, val in data.items():
        display_key = _FIELD_RENAMES.get(key, key)
        if key in _CODE_LABELS and isinstance(val, (int, float)):
            label = _CODE_LABELS[key].get(int(val))
            if label:
                out[display_key] = f"{label} ({int(val)})"
                continue
        out[display_key] = val
    return out

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

#: Module-level few-shot examples injected into every prompt.
#: Each entry must have an ``"input"`` key (dict of feature values) and a
#: ``"label"`` key whose value is either ``"normal"`` or ``"anomaly"``.
#: Ratio: 3 normal, 7 anomaly — biased toward anomalies to improve recall.
FEW_SHOT_EXAMPLES: list[dict[str, Any]] = [
    # --- Normal examples (3) ---
    {
        "input": {
            "age": 43, "gender": 1600001, "family_relation": 1700001,
            "marage_status": 10600002, "nationality": 1800001,
            "q_301": 10500025, "q_602_val": 30000,
            "cut_5_total": 40, "act_1_total": 38,
        },
        "label": "normal",
    },
    {
        "input": {
            "age": 35, "gender": 1600002, "family_relation": 1700021,
            "marage_status": 10600002, "nationality": 1800001,
            "q_301": 10500019, "q_602_val": 3000,
            "cut_5_total": 30, "act_1_total": 28,
        },
        "label": "normal",
    },
    {
        "input": {
            "age": 55, "gender": 1600001, "family_relation": 1700001,
            "marage_status": 10600002, "nationality": 1800001,
            "q_301": 10500031, "q_602_val": 2000,
            "cut_5_total": 48, "act_1_total": 50,
        },
        "label": "normal",
    },
    # --- Anomaly examples (7) ---
    {
        "input": {
            "age": 35, "gender": 1600001, "family_relation": 1700001,
            "marage_status": 10600002, "nationality": 1800001,
            "q_301": 10500031, "q_602_val": 45000,
            "cut_5_total": 40, "act_1_total": 40,
        },
        "label": "anomaly",
    },
    {
        "input": {
            "age": 50, "gender": 1600001, "family_relation": 1700001,
            "marage_status": 10600002, "nationality": 1800001,
            "q_301": 10500010, "q_602_val": 500,
            "cut_5_total": 40, "act_1_total": 38,
        },
        "label": "anomaly",
    },
    {
        "input": {
            "age": 30, "gender": 1600001, "family_relation": 1700022,
            "marage_status": 10600001, "nationality": 1800001,
            "q_301": 10500019, "q_602_val": 5000,
            "cut_5_total": 84, "act_1_total": 0,
        },
        "label": "anomaly",
    },
    {
        "input": {
            "age": 38, "gender": 1600001, "family_relation": 1700001,
            "marage_status": 10600002, "nationality": 1800001,
            "q_301": 10500025, "q_602_val": 49000,
            "cut_5_total": 1, "act_1_total": 1,
        },
        "label": "anomaly",
    },
    {
        "input": {
            "age": 15, "gender": 1600001, "family_relation": 1700022,
            "marage_status": 10600001, "nationality": 1800001,
            "q_301": 10500031, "q_602_val": 5000,
            "cut_5_total": 80, "act_1_total": 80,
        },
        "label": "anomaly",
    },
    {
        "input": {
            "age": 70, "gender": 1600001, "family_relation": 1700001,
            "marage_status": 10600004, "nationality": 1800001,
            "q_301": 10500031, "q_602_val": 500,
            "cut_5_total": 84, "act_1_total": 84,
        },
        "label": "anomaly",
    },
    {
        "input": {
            "age": 45, "gender": 1600002, "family_relation": 1700021,
            "marage_status": 10600002, "nationality": 1800001,
            "q_301": 10500023, "q_602_val": 10000,
            "cut_5_total": 1, "act_1_total": 84,
        },
        "label": "anomaly",
    },
]


# ---------------------------------------------------------------------------
# Prompt scaffolding
# ---------------------------------------------------------------------------

_CONFIRM_INSTRUCTIONS: str = (
    "You are a data-quality reviewer for Saudi Labour Force Survey (LFS) records. "
    "A statistical model flagged the record below for review. "
    "Statistical models often produce FALSE POSITIVES — many flagged records are actually fine. "
    "Your job is to determine if this record has a REAL problem or is a false alarm.\n\n"

    "== Field codes ==\n"
    "gender: 1600001=Male, 1600002=Female\n"
    "family_relation: 1700001=Head, 1700021=Spouse, 1700022=Son/Daughter\n"
    "marage_status: 10600001=Never married, 10600002=Married, 10600003=Divorced, "
    "10600004=Widowed\n"
    "q_301 (education): 10500031=No formal edu, 10500003=Primary, "
    "10500017=Intermediate, 10500019=Secondary, 10500023=Diploma, "
    "10500025=Bachelor's, 10500009=Master's, 10500010=PhD\n"
    "q_602_val=exact monthly salary in SAR (e.g. 500 means 500 SAR, NOT thousands), "
    "cut_5_total=usual weekly hours, "
    "act_1_total=actual weekly hours\n\n"

    "== Salary ranges by education ==\n"
    "No edu/Primary: 1,500-5,000 | Secondary: 3,000-12,000 | "
    "Diploma: 4,000-18,000 | Bachelor's: 5,000-25,000 | "
    "Master's: 8,000-35,000 | PhD: 12,000-45,000\n\n"

    "== Rules ==\n"
    "- is_anomaly=TRUE only if salary is clearly outside the education range, "
    "OR hours gap (usual vs actual) > 30, OR age is impossible for the education level\n"
    "- is_anomaly=FALSE if all values are within plausible ranges, even if unusual. "
    "A 55yo with no education earning 4,000 SAR at 48 hrs/week is NORMAL. "
    "A 22yo bachelor earning 8,000 SAR at 40 hrs/week is NORMAL. "
    "Do NOT flag records just because the statistical model did.\n\n"

    "== Severity ==\n"
    "- hard_error: impossible data (age 12 head of household, spouse not married, "
    "education impossible for age). These are definite data entry errors.\n"
    "- warning: suspicious but plausible (unusual salary for education level, "
    "high/low working hours, salary outliers). These need human review.\n\n"

    "Respond with JSON only:\n"
    '{"is_anomaly": <true|false>, "score": <float 0.0-1.0>, '
    '"severity": "<hard_error|warning>", "explanation": "<brief reason>"}'
)

_EXPLAIN_INSTRUCTIONS: str = (
    "أنت مساعد لتفسير الشذوذ في بيانات مسح القوى العاملة السعودي.\n"
    "السجل التالي تم تصنيفه كشاذ. اشرح المشكلة بالعربية.\n\n"

    "== رموز ��لحقول ==\n"
    "gender: 1600001=ذكر, 1600002=أنثى\n"
    "family_relation: 1700001=رئيس الأسرة, 1700021=زوج/زوجة, 1700022=ابن/ابنة, 1700010=عمالة منزلية\n"
    "marage_status: 10600001=لم ��سبق الزواج, 10600002=مت��وج, 10600003=مطلق, 10600004=أرمل\n"
    "q_301: 10500031=ب��ون تعليم, 10500003=ابتدائي, 10500017=متوسط, 10500019=ثانوي, "
    "10500023=دبلوم, 10500025=بكالوريوس, 10500009=ماجستير, 10500010=دكتوراه\n"
    "q_602_val=الراتب الشه��ي (ريال), cut_5_total=ساعات العمل المعتادة, "
    "act_1_total=ساعات العمل الفعلية\n\n"

    "== مثال ==\n"
    "السجل: ذكر، 35 سنة، بدون تعليم، راتب 45,000 ريال، 40 ساعة عمل\n"
    '{"explanation": "يوجد تناقض واضح بين المستوى التعليمي والراتب. '
    "الشخص مسجل بدون تعليم رسمي لكن راتبه 45,000 ريال شهرياً، "
    'بينما المتوقع لهذا المستوى التعليمي هو 1,500-5,000 ريال. '
    'يُحتمل أن يكون هناك خطأ في إدخال البيانات."}\n\n'

    "اكتب التفسير بنفس الأسلوب: حدد المشكلة، اذكر القيم الفعلية والمتوقعة، "
    "واقترح السبب المحتمل.\n\n"

    "أجب بـ JSON فقط:\n"
    '{"explanation": "<2-3 جمل بالعربية>"}'
)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class LLMDetector(BaseDetector):
    """Few-shot anomaly detector backed by an LLM running under Ollama.

    Attributes:
        _base_url: Ollama server base URL (trailing slash stripped).
        _model: Ollama model tag to request (e.g. ``"qwen3.5"``).
        _transport: Optional httpx transport override, used in tests to avoid
            live network calls.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "qwen3.5",
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Initialise the detector.

        Args:
            base_url: Ollama server base URL, e.g. ``"http://localhost:11434"``.
            model: Model tag to request from Ollama.
            transport: Optional httpx async transport override.  Pass an
                :class:`httpx.MockTransport` instance in tests to avoid
                requiring a live Ollama server.
        """
        self._base_url: str = base_url.rstrip("/")
        self._model: str = model
        self._transport: httpx.AsyncBaseTransport | None = transport

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def detect(
        self,
        data: dict[str, Any],
        *,
        svm_score: float | None = None,
        ge_score: float | None = None,
        findings: list[str] | None = None,
    ) -> StrategyResult:
        """Send a prompt to the LLM and parse the anomaly verdict.

        When *svm_score* or *ge_score* is provided the LLM operates in
        **confirmation mode**: the record has already been flagged by
        upstream detectors and the LLM is asked to confirm or deny the
        flag.  Without context it falls back to independent few-shot
        classification.

        Args:
            data: Flat or nested dictionary of feature values to classify.
            svm_score: Optional SVM anomaly score (0-1) from upstream.
            ge_score: Optional GE anomaly score from upstream.

        Returns:
            A :class:`~backend.models.schemas.StrategyResult` for the LLM
            strategy.
        """
        if svm_score is not None or ge_score is not None:
            prompt = self._build_confirm_prompt(data, svm_score, ge_score, findings)
        else:
            prompt = self._build_prompt(data)

        try:
            raw_text = await self._call_ollama(prompt)
            parsed = self._parse_response(raw_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LLMDetector: failed to obtain or parse Ollama response (%s: %s). "
                "Returning fail-safe result.",
                type(exc).__name__,
                exc,
            )
            return StrategyResult(
                strategy=StrategyName.llm,
                is_anomaly=False,
                score=0.0,
                explanation="Fail-safe: LLM unavailable or response unparseable.",
            )

        is_anomaly: bool = bool(parsed.get("is_anomaly", False))
        score: float = max(0.0, min(1.0, float(parsed.get("score") or 0.0)))
        explanation: str = str(parsed.get("explanation", ""))

        return StrategyResult(
            strategy=StrategyName.llm,
            is_anomaly=is_anomaly,
            score=round(score, 4),
            explanation=explanation,
            raw=parsed,
        )

    async def explain(
        self,
        data: dict[str, Any],
        ge_score: float | None = None,
        ge_failed_rules: int = 0,
        svm_score: float | None = None,
    ) -> str:
        """Ask the LLM to explain why a record is anomalous.

        Called by the detection service after GE/SVM have flagged the record.
        Returns a human-readable Arabic explanation.

        Args:
            data: The original data record.
            ge_score: GE anomaly score (fraction of failed rules).
            ge_failed_rules: Number of GE rules that failed.
            svm_score: SVM anomaly score.

        Returns:
            A string explanation in Arabic, or a fallback message on failure.
        """
        prompt = self._build_explain_prompt(data, ge_score, ge_failed_rules, svm_score)
        try:
            raw_text = await self._call_ollama(prompt)
            parsed = self._parse_response(raw_text)
            return str(parsed.get("explanation", raw_text))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LLMDetector.explain failed (%s: %s). Returning fallback.",
                type(exc).__name__,
                exc,
            )
            return "تعذر توليد التفسير من نموذج الذكاء الاصطناعي."

    async def health_check(self) -> bool:
        """Ping the Ollama server to verify it is reachable.

        Sends a ``GET /api/tags`` request and returns ``True`` on any 2xx
        response.  Returns ``False`` on network errors or non-2xx status.

        Returns:
            ``True`` if Ollama is reachable and responsive.
        """
        kwargs: dict[str, Any] = {}
        if self._transport is not None:
            kwargs["transport"] = self._transport

        try:
            async with httpx.AsyncClient(**kwargs) as client:
                response = await client.get(
                    f"{self._base_url}/api/tags",
                    timeout=5.0,
                )
                return response.is_success
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, data: dict[str, Any]) -> str:
        """Construct the few-shot prompt string from examples and the candidate.

        Serialises each example from :data:`FEW_SHOT_EXAMPLES` as an
        ``input`` / ``label`` pair, then appends the candidate ``data`` for
        classification.

        Args:
            data: The candidate data record to classify.

        Returns:
            A fully-formed prompt string ready to send to the LLM.
        """
        lines: list[str] = [_CONFIRM_INSTRUCTIONS, ""]
        lines.append("--- Few-shot examples ---")
        for i, example in enumerate(FEW_SHOT_EXAMPLES, start=1):
            lines.append(f"Example {i}:")
            lines.append(f"  Input: {json.dumps(example['input'])}")
            lines.append(f"  Label: {example['label']}")
        lines.append("")
        lines.append("--- Record to classify ---")
        lines.append(f"Input: {json.dumps(_humanise_record(data))}")
        lines.append("")
        lines.append("Respond with JSON only.")
        return "\n".join(lines)

    def _build_explain_prompt(
        self,
        data: dict[str, Any],
        ge_score: float | None,
        ge_failed_rules: int,
        svm_score: float | None,
    ) -> str:
        """Build a prompt asking the LLM to explain detected anomalies."""
        lines: list[str] = [_EXPLAIN_INSTRUCTIONS, ""]
        lines.append("--- Flagged Record ---")
        lines.append(f"Data: {json.dumps(_humanise_record(data), ensure_ascii=False)}")
        lines.append("")
        lines.append("--- Detection Findings ---")
        if ge_score is not None:
            lines.append(
                f"Business rules (GE): {ge_failed_rules} rule(s) violated, "
                f"score {ge_score:.2%}"
            )
        if svm_score is not None:
            lines.append(f"Statistical model (SVM): anomaly score {svm_score:.2%}")
        lines.append("")
        lines.append("Respond with JSON only.")
        return "\n".join(lines)

    def _build_confirm_prompt(
        self,
        data: dict[str, Any],
        svm_score: float | None,
        ge_score: float | None,
        findings: list[str] | None = None,
    ) -> str:
        """Build a confirmation prompt with upstream detector context."""
        lines: list[str] = [_CONFIRM_INSTRUCTIONS, ""]
        lines.append("--- Flagged Record ---")
        human_data = _humanise_record(data)
        lines.append(f"Data: {json.dumps(human_data)}")
        salary = data.get("q_602_val")
        if salary is not None:
            lines.append(f"NOTE: The salary field q_602_val={salary} means exactly {salary} SAR per month.")
        lines.append("")
        lines.append("--- Why it was flagged ---")
        if findings:
            lines.append("Business rule violations found:")
            for f in findings:
                lines.append(f"  - {f}")
        if svm_score is not None:
            pct = int(svm_score * 100)
            lines.append(f"Statistical model confidence: {pct}% likely anomalous")
        if ge_score is not None and ge_score > 0:
            lines.append(f"Business rules score: {ge_score:.2%}")
        lines.append("")
        lines.append("Summarize ALL the problems found in your explanation. Respond with JSON only.")
        return "\n".join(lines)

    async def _call_ollama(self, prompt: str) -> str:
        """POST *prompt* to ``/api/generate`` and return the model's response text.

        Args:
            prompt: The fully-formatted prompt string.

        Returns:
            The raw text content from the ``"response"`` field of the Ollama
            API response body.

        Raises:
            httpx.HTTPStatusError: On non-2xx HTTP responses.
            httpx.RequestError: On connection or timeout errors.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 2048},
        }
        kwargs: dict[str, Any] = {}
        if self._transport is not None:
            kwargs["transport"] = self._transport

        async with httpx.AsyncClient(**kwargs) as client:
            response = await client.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            return str(body.get("response", ""))

    def _parse_response(self, raw: str) -> dict[str, Any]:
        """Extract the JSON payload from the model's response text.

        Attempts a direct :func:`json.loads` first.  If that fails, falls back
        to extracting the first ``{...}`` block via a regex so that models that
        wrap their JSON in prose are still handled correctly.

        Args:
            raw: The raw text returned by the LLM.

        Returns:
            A dictionary containing at minimum ``"is_anomaly"``, ``"score"``,
            and ``"explanation"`` keys (values may be absent / defaulted in
            :meth:`detect`).

        Raises:
            ValueError: If no valid JSON object can be extracted from *raw*.
        """
        # Strip <think>...</think> blocks (Qwen3.5 chain-of-thought)
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Strip markdown code fences (```json ... ```)
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        # Fast path: the whole cleaned string is valid JSON.
        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Fallback: grab the first {...} block (handles leading/trailing prose).
        match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"Cannot extract a JSON object from LLM response: {raw!r}"
        )
