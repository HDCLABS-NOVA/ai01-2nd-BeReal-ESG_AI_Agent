from __future__ import annotations

import json
import logging
import math
import os
import re
import textwrap
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .utils import clamp, sentence_tokenize, to_csv

LOGGER = logging.getLogger(__name__)

try:  # Optional sentence embedding
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

try:  # XLSX export
    from openpyxl import Workbook
except ImportError:  # pragma: no cover - optional dependency
    Workbook = None

try:  # PDF export
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as pdf_canvas
except ImportError:  # pragma: no cover - optional dependency
    pdf_canvas = None
    A4 = None

try:  # LLM validation/extraction
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None
    ChatPromptTemplate = None


@dataclass(frozen=True)
class ScoreBehavior:
    scoring_mode: str = "additive"
    zero_tolerance_keywords: Tuple[str, ...] = field(default_factory=tuple)
    critical: bool = False
    metric_pattern: str | None = None
    requires_evidence: bool = False
    bonus_cap: float | None = None
    penalty_cap: float | None = None
    signal_cap: int = 3
    industry_overrides: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRow:
    area: str
    item: str
    criterion: str
    weight: float
    base_score: float
    positive_signals: Tuple[Tuple[str, float], ...]
    negative_signals: Tuple[Tuple[str, float], ...]
    evidence_keywords: Tuple[str, ...]
    synonyms: Tuple[str, ...]
    behavior: ScoreBehavior


@dataclass(frozen=True)
class GradeThreshold:
    grade: str
    min_ratio: float
    label: str


@dataclass(frozen=True)
class TemplateBundle:
    name: str
    version: str
    tags: Tuple[str, ...]
    rows: Tuple[EvaluationRow, ...]
    grade_thresholds: Tuple[GradeThreshold, ...]
    positive_signals: Tuple[Tuple[str, float], ...]
    negative_signals: Tuple[Tuple[str, float], ...]


@dataclass
class SupplierEvaluationRequest:
    supplier: str
    industry: str
    context: str
    documents: List[str] | None = None


@dataclass
class EvidenceMatch:
    sentence: str
    score: float
    validated: bool
    validated_reason: str | None = None


BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "data" / "supplier_eval_template.json"
TEMPLATE_DIR = BASE_DIR / "data" / "supplier_templates"
BENCHMARK_DIR = BASE_DIR / "data" / "companies"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
MAX_CONTEXT_SENTENCES = 250


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z가-힣]+", "-", value.strip())
    cleaned = re.sub(r"-+", "-", cleaned)
    return cleaned.strip("-").lower() or "supplier"


def _normalize(text: str) -> str:
    return text.strip().lower()


def _read_json(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _load_templates() -> Tuple[TemplateBundle, ...]:
    candidate_paths: List[Path] = []
    if CONFIG_PATH.exists():
        candidate_paths.append(CONFIG_PATH)
    if TEMPLATE_DIR.exists():
        candidate_paths.extend(sorted(TEMPLATE_DIR.glob("*.json")))
    bundles: List[TemplateBundle] = []
    for path in candidate_paths:
        try:
            data = _read_json(path)
            bundles.append(_parse_template(data, path.name))
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("템플릿 로드 실패 %s: %s", path, exc)
    if not bundles:
        raise RuntimeError("Supplier 평가 템플릿을 찾을 수 없습니다.")
    return tuple(bundles)


def _parse_template(data: Dict[str, object], source_name: str) -> TemplateBundle:
    version = str(data.get("version", "Supplier Template"))
    name = str(data.get("name", source_name))
    tags = tuple(_normalize(tag) for tag in data.get("industry_tags", []))
    grade_thresholds = [
        GradeThreshold(
            grade=str(entry.get("grade", "C")),
            min_ratio=float(entry.get("min_ratio", 0.0)),
            label=str(entry.get("label", "")),
        )
        for entry in data.get("grade_thresholds", [])
    ]
    grade_thresholds.sort(key=lambda t: t.min_ratio, reverse=True)
    if not grade_thresholds:
        grade_thresholds.append(GradeThreshold("C", 0.0, "기본"))

    rows: List[EvaluationRow] = []
    for area in data.get("areas", []):
        area_name = area.get("name", "미분류")
        area_weight = float(area.get("weight", 1.0))
        for item in area.get("items", []):
            item_weight = float(item.get("weight", 1.0))
            positive = tuple(
                (entry.get("keyword", "").lower(), float(entry.get("impact", 1.0)))
                for entry in item.get("positive_signals", [])
                if entry.get("keyword")
            )
            negative = tuple(
                (entry.get("keyword", "").lower(), float(entry.get("impact", 1.0)))
                for entry in item.get("negative_signals", [])
                if entry.get("keyword")
            )
            evidence_keywords = tuple(keyword.lower() for keyword in item.get("evidence_keywords", []))
            synonyms = tuple(keyword.lower() for keyword in item.get("synonyms", []))
            behavior = ScoreBehavior(
                scoring_mode=item.get("scoring_mode", "additive"),
                zero_tolerance_keywords=tuple(
                    keyword.lower() for keyword in item.get("zero_tolerance_keywords", [])
                ),
                critical=bool(item.get("critical", False)),
                metric_pattern=item.get("metric_pattern"),
                requires_evidence=bool(item.get("requires_evidence", False)),
                bonus_cap=item.get("bonus_cap"),
                penalty_cap=item.get("penalty_cap"),
                signal_cap=int(item.get("signal_cap", 3)),
                industry_overrides={_normalize(k): float(v) for k, v in item.get("area_overrides", {}).items()},
            )
            rows.append(
                EvaluationRow(
                    area=area_name,
                    item=item.get("name", ""),
                    criterion=item.get("criterion", ""),
                    weight=area_weight * item_weight,
                    base_score=float(item.get("base_score", 3.0)),
                    positive_signals=positive,
                    negative_signals=negative,
                    evidence_keywords=evidence_keywords,
                    synonyms=synonyms,
                    behavior=behavior,
                )
            )
    return TemplateBundle(
        name=name,
        version=version,
        tags=tags,
        rows=tuple(rows),
        grade_thresholds=tuple(grade_thresholds),
        positive_signals=tuple(
            (keyword.lower(), float(value))
            for keyword, value in data.get("global_positive_signals", {}).items()
        ),
        negative_signals=tuple(
            (keyword.lower(), float(value))
            for keyword, value in data.get("global_negative_signals", {}).items()
        ),
    )


def _select_template(industry: str) -> TemplateBundle:
    normalized = _normalize(industry or "")
    for template in _load_templates():
        if not template.tags:
            continue
        if any(tag in normalized for tag in template.tags):
            return template
    return _load_templates()[0]


@lru_cache(maxsize=1)
def _benchmark_companies() -> Tuple[str, ...]:
    if not BENCHMARK_DIR.exists():
        return tuple()
    names: List[str] = []
    for path in sorted(BENCHMARK_DIR.glob("*.pdf")):
        stem = re.sub(r"[_-]?ESG.*", "", path.stem, flags=re.IGNORECASE).strip()
        names.append(stem or path.stem)
    return tuple(dict.fromkeys(names))


def _build_context_chunks(request: SupplierEvaluationRequest) -> Tuple[str, ...]:
    sources: List[str] = []
    if request.documents:
        sources.extend(request.documents)
    if request.context:
        sources.append(request.context)
    sentences: List[str] = []
    for source in sources:
        sentences.extend(sentence_tokenize(source))
    cleaned = [sentence.strip() for sentence in sentences if sentence and sentence.strip()]
    trimmed = cleaned[:MAX_CONTEXT_SENTENCES]
    return tuple(trimmed) if trimmed else (request.context.strip(),)


@lru_cache(maxsize=1)
def _embedding_model() -> SentenceTransformer | None:
    if SentenceTransformer is None:
        return None
    model_name = os.getenv(
        "SUPPLIER_EVAL_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.error("임베딩 모델 로드 실패 (%s): %s", model_name, exc)
        return None


def _embed_sentences(sentences: Sequence[str]) -> np.ndarray | None:
    model = _embedding_model()
    if not model or not sentences:
        return None
    return model.encode(list(sentences), normalize_embeddings=True)


class EvidenceValidator:
    def __init__(self) -> None:
        model_name = os.getenv("SUPPLIER_EVAL_VALIDATOR_MODEL")
        if model_name and ChatOpenAI and ChatPromptTemplate:
            self.llm = ChatOpenAI(model=model_name, temperature=0)
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "주어진 문장이 평가항목의 근거인지 yes/no로만 답변하세요.",
                    ),
                    (
                        "human",
                        "항목: {item}\n기준: {criterion}\n문장: {sentence}\n답변:",
                    ),
                ]
            )
        else:
            self.llm = None
            self.prompt = None

    def is_valid(self, row: EvaluationRow, sentence: str) -> Tuple[bool, str | None]:
        if not sentence:
            return False, None
        lowered = sentence.lower()
        heuristic_tokens = [row.item, row.area, *row.evidence_keywords, *row.synonyms]
        heuristic = any(token and token in lowered for token in heuristic_tokens)
        if not self.llm:
            return heuristic, "heuristic"
        try:
            messages = self.prompt.format_prompt(item=row.item, criterion=row.criterion, sentence=sentence).to_messages()
            response = self.llm.invoke(messages)
            answer = (getattr(response, "content", "") or "").strip().lower()
            if answer.startswith("yes") or answer.startswith("y"):
                return True, answer
            if answer.startswith("no"):
                return False, answer
            return heuristic, answer
        except Exception:
            return heuristic, "heuristic"


class EvidenceMatcher:
    def __init__(self, sentences: Sequence[str]) -> None:
        self.sentences = sentences
        self.embeddings = _embed_sentences(sentences)

    @staticmethod
    def build_query(row: EvaluationRow) -> str:
        tokens = [row.item, *row.synonyms]
        tokens = [token for token in tokens if token and len(token) >= 2]
        return " ".join(tokens)

    def match(self, query: str, top_k: int = 2) -> List[Tuple[str, float]]:
        if not self.sentences:
            return []
        if self.embeddings is None:
            return self._lexical_match(query, top_k)
        model = _embedding_model()
        if model is None:
            return self._lexical_match(query, top_k)
        query_vec = model.encode([query], normalize_embeddings=True)[0]
        sims = np.dot(self.embeddings, query_vec)
        top_indices = np.argsort(-sims)[:top_k]
        return [(self.sentences[idx], float(sims[idx])) for idx in top_indices]

    def _lexical_match(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        tokens = [token for token in re.split(r"\s+", query.lower()) if len(token) >= 2]
        scores: List[Tuple[str, float]] = []
        for sentence in self.sentences:
            lowered = sentence.lower()
            score = sum(lowered.count(token) for token in tokens)
            if score:
                scores.append((sentence, float(score)))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]


class SignalExtractor:
    def __init__(self, template: TemplateBundle) -> None:
        self.template = template
        model_name = os.getenv("SUPPLIER_EVAL_SIGNAL_MODEL")
        if model_name and ChatOpenAI and ChatPromptTemplate:
            self.llm = ChatOpenAI(model=model_name, temperature=0)
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "문맥에서 ESG 긍정/부정 신호를 JSON으로 요약하세요. keys: positive, negative",
                    ),
                    (
                        "human",
                        "문맥:\n{context}\n\n응답 형식 예시: {\"positive\": [\"ISO14001 인증\"], \"negative\": [\"중대재해\"]}",
                    ),
                ]
            )
        else:
            self.llm = None
            self.prompt = None

    def extract(self, context: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        if self.llm and self.prompt:
            try:
                messages = self.prompt.format_prompt(context=context[:2000]).to_messages()
                response = self.llm.invoke(messages)
                parsed = json.loads(getattr(response, "content", "") or "{}")
                positives = {
                    phrase.lower(): self._lookup_signal_value(phrase.lower(), True)
                    for phrase in parsed.get("positive", [])
                }
                negatives = {
                    phrase.lower(): self._lookup_signal_value(phrase.lower(), False)
                    for phrase in parsed.get("negative", [])
                }
                positives = {k: v for k, v in positives.items() if v}
                negatives = {k: v for k, v in negatives.items() if v}
                if positives or negatives:
                    return positives, negatives
            except Exception:
                pass
        return self._dictionary_scan(context.lower())

    def _lookup_signal_value(self, keyword: str, positive: bool) -> float:
        signals = self.template.positive_signals if positive else self.template.negative_signals
        best_value = 0.0
        best_ratio = 0.0
        for key, value in signals:
            if not key:
                continue
            if key in keyword or keyword in key:
                return value
            ratio = SequenceMatcher(None, key, keyword).ratio()
            if ratio > 0.85 and ratio > best_ratio:
                best_ratio = ratio
                best_value = value
        return best_value

    def _dictionary_scan(self, lowered: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        positives = {
            keyword: impact
            for keyword, impact in self.template.positive_signals
            if keyword and keyword in lowered
        }
        negatives = {
            keyword: impact
            for keyword, impact in self.template.negative_signals
            if keyword and keyword in lowered
        }
        return positives, negatives


def generate_template_csv(supplier: str, industry: str) -> str:
    template = _select_template(industry)
    rows = template.rows
    benchmarks = ", ".join(_benchmark_companies()) or "벤치마크 미등록"
    header_meta = [
        "버전",
        template.version,
        "협력사명",
        supplier,
        "업종",
        industry,
        "벤치마크",
        benchmarks,
        "템플릿",
        template.name,
    ]
    csv_body = to_csv(
        ["영역", "평가항목", "평가기준", "배점(0~5)", "비고"],
        [
            [
                row.area,
                row.item,
                row.criterion,
                "0~5",
                f"가중치 {row.weight:.2f}" if row.weight != 1.0 else "",
            ]
            for row in rows
        ],
    )
    return f"{','.join(header_meta)}\n{csv_body}"


def _apply_signals(
    base: float,
    signals: Tuple[Tuple[str, float], ...],
    lowered_context: str,
    tag: str,
    behavior: ScoreBehavior,
    rationale: List[str],
) -> float:
    total_delta = 0.0
    for keyword, impact in signals:
        if not keyword:
            continue
        occurrences = lowered_context.count(keyword)
        if not occurrences:
            continue
        capped = min(occurrences, behavior.signal_cap)
        delta = capped * (impact if tag == "positive" else -impact)
        total_delta += delta
        direction = "+" if delta > 0 else "-"
        rationale.append(f"{keyword} {direction}{abs(delta):.1f}")
    if behavior.bonus_cap is not None and total_delta > behavior.bonus_cap:
        total_delta = behavior.bonus_cap
    if behavior.penalty_cap is not None and total_delta < -behavior.penalty_cap:
        total_delta = -behavior.penalty_cap
    return base + total_delta


def _extract_metric_value(pattern: str, context: str) -> float | None:
    matches = re.findall(pattern, context, flags=re.IGNORECASE)
    values: List[float] = []
    for match in matches:
        if isinstance(match, tuple):
            match = match[0]
        try:
            values.append(float(str(match).replace(",", "")))
        except ValueError:
            continue
    return max(values) if values else None


def _score_row(
    row: EvaluationRow,
    lowered_context: str,
    evidence: EvidenceMatch | None,
) -> Tuple[float, List[str], bool]:
    behavior = row.behavior
    rationale: List[str] = []
    critical_triggered = False
    base = row.base_score

    if behavior.zero_tolerance_keywords:
        if any(keyword in lowered_context for keyword in behavior.zero_tolerance_keywords):
            rationale.append("제로 톨러런스 키워드 감지")
            critical_triggered = behavior.critical
            return 0.0, rationale, critical_triggered

    if behavior.scoring_mode == "log_scale" and behavior.metric_pattern:
        metric_value = _extract_metric_value(behavior.metric_pattern, lowered_context)
        if metric_value is not None:
            delta = math.log1p(metric_value)
            rationale.append(f"log scale +{delta:.1f} (metric {metric_value})")
            base += delta

    base = _apply_signals(base, row.negative_signals, lowered_context, "negative", behavior, rationale)
    base = _apply_signals(base, row.positive_signals, lowered_context, "positive", behavior, rationale)

    if behavior.requires_evidence and not (evidence and evidence.validated):
        rationale.append("근거 미인증 → -1.0")
        base -= 1.0

    if evidence and evidence.validated:
        rationale.append(f"근거: {evidence.sentence.strip()[:120]}")
        if evidence.validated_reason:
            rationale.append(f"근거 검증: {evidence.validated_reason}")

    score = round(clamp(base), 2)
    if not rationale:
        rationale.append("직접 근거 부족 - 기본점수")
    return score, rationale, critical_triggered


def _match_global_signals(template: TemplateBundle, context: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    extractor = SignalExtractor(template)
    return extractor.extract(context)


def _build_db_payload(
    request: SupplierEvaluationRequest,
    template: TemplateBundle,
    summary: str,
    score_rows: List[Dict[str, object]],
    grade_info: Dict[str, str],
) -> Dict[str, object]:
    return {
        "supplier": request.supplier,
        "industry": request.industry,
        "template": template.name,
        "version": template.version,
        "summary": summary,
        "grade": grade_info["grade"],
        "grade_label": grade_info["label"],
        "grade_note": grade_info.get("note", ""),
        "rows": score_rows,
    }


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _render_xlsx(
    slug: str,
    score_rows: List[Dict[str, object]],
    summary: str,
    grade_info: Dict[str, str],
) -> str | None:
    if Workbook is None:
        return None
    _ensure_output_dir()
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Scorecard"
    sheet.append(["영역", "항목", "배점", "가중치", "근거"])
    for row in score_rows:
        sheet.append(
            [
                row["area"],
                row["item"],
                row["score"],
                row["weight"],
                textwrap.shorten(row["rationale"], width=80),
            ]
        )
    sheet.append([])
    sheet.append([summary])
    sheet.append([f"등급 {grade_info['grade']} ({grade_info['label']})"])
    path = OUTPUT_DIR / f"{slug}_scorecard.xlsx"
    workbook.save(path)
    return str(path)


def _render_pdf(
    slug: str,
    request: SupplierEvaluationRequest,
    summary: str,
    grade_info: Dict[str, str],
    risks: List[str],
    strengths: List[str],
) -> str | None:
    if pdf_canvas is None or A4 is None:
        return None
    _ensure_output_dir()
    path = OUTPUT_DIR / f"{slug}_report.pdf"
    c = pdf_canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    text = c.beginText(40, height - 60)
    text.setFont("Helvetica", 11)
    raw_lines = [
        f"협력사 ESG 평가 보고서",
        f"협력사: {request.supplier}",
        f"업종: {request.industry}",
        summary,
        f"등급: {grade_info['grade']} ({grade_info['label']})",
        f"핵심 리스크: {', '.join(risks[:3]) if risks else 'N/A'}",
        f"강점 영역: {', '.join(strengths[:3]) if strengths else 'N/A'}",
    ]
    for line in raw_lines:
        wrapped = textwrap.wrap(line, width=80) or [line]
        for chunk in wrapped:
            text.textLine(chunk)
    c.drawText(text)
    c.showPage()
    c.save()
    return str(path)


def score_supplier(request: SupplierEvaluationRequest) -> Dict[str, object]:
    template = _select_template(request.industry)
    industry_key = _normalize(request.industry or "")
    context_chunks = _build_context_chunks(request)
    lowered_context = " \n".join(context_chunks).lower()
    matcher = EvidenceMatcher(context_chunks)
    validator = EvidenceValidator()

    rows_for_csv: List[Sequence[str]] = []
    weighted_total = 0.0
    max_score = 0.0
    strengths: List[str] = []
    risks: List[str] = []
    detailed_rows: List[Dict[str, object]] = []
    critical_triggered = False

    for row in template.rows:
        query = EvidenceMatcher.build_query(row)
        matches = matcher.match(query, top_k=2)
        evidence_match: EvidenceMatch | None = None
        for sentence, similarity in matches:
            is_valid, reason = validator.is_valid(row, sentence)
            evidence_match = EvidenceMatch(
                sentence=sentence,
                score=similarity,
                validated=is_valid,
                validated_reason=reason,
            )
            if is_valid:
                break
        score, rationale, row_critical = _score_row(row, lowered_context, evidence_match)
        weight = row.weight * row.behavior.industry_overrides.get(industry_key, 1.0)
        weighted_total += score * weight
        max_score += 5 * weight
        formatted_score = f"{score:.2f}" if not score.is_integer() else str(int(score))
        rationale_text = " / ".join(rationale)
        rows_for_csv.append((row.area, row.item, row.criterion, formatted_score, rationale_text))
        detailed_rows.append(
            {
                "area": row.area,
                "item": row.item,
                "score": score,
                "weight": round(weight, 3),
                "rationale": rationale_text,
                "evidence": evidence_match.sentence if evidence_match else "",
                "evidence_reason": evidence_match.validated_reason if evidence_match else "",
            }
        )
        if score <= 2.5:
            risks.append(f"{row.area}-{row.item} ({score:.1f})")
        if score >= 4.0:
            strengths.append(f"{row.area}-{row.item} ({score:.1f})")
        critical_triggered = critical_triggered or row_critical

    positives, negatives = _match_global_signals(template, lowered_context)
    global_delta = sum(positives.values()) - sum(negatives.values())
    weighted_total = max(0.0, min(max_score, weighted_total + global_delta))

    csv_score = to_csv(["영역", "평가항목", "평가기준", "배점", "근거"], rows_for_csv)
    grade_info = grade_supplier(template, weighted_total, max_score, critical_triggered)
    summary = (
        f"가중 총점 {weighted_total:.1f}점 / {max_score:.1f}점 → 등급 {grade_info['grade']}"
        f" ({grade_info['label']})"
    )
    if grade_info.get("note"):
        summary += f" | {grade_info['note']}"
    global_notes: List[str] = []
    if positives:
        global_notes.append(
            "긍정 신호: " + ", ".join(f"{keyword} (+{impact:.1f})" for keyword, impact in positives.items())
        )
    if negatives:
        global_notes.append(
            "부정 신호: " + ", ".join(f"{keyword} (-{impact:.1f})" for keyword, impact in negatives.items())
        )

    slug = _slugify(f"{request.supplier}_{request.industry}")
    xlsx_path = _render_xlsx(slug, detailed_rows, summary, grade_info)
    pdf_path = _render_pdf(slug, request, summary, grade_info, risks, strengths)
    db_payload = _build_db_payload(request, template, summary, detailed_rows, grade_info)

    return {
        "csv": csv_score,
        "summary": summary,
        "total": weighted_total,
        "max_score": max_score,
        "grade": grade_info["grade"],
        "grade_label": grade_info["label"],
        "grade_note": grade_info.get("note", ""),
        "global_notes": global_notes,
        "risks": risks,
        "strengths": strengths,
        "rows": detailed_rows,
        "benchmarks": list(_benchmark_companies()),
        "template": template.name,
        "version": template.version,
        "xlsx_path": xlsx_path,
        "pdf_path": pdf_path,
        "db_payload": db_payload,
    }


def grade_supplier(
    template: TemplateBundle,
    score: float,
    max_score: float,
    critical: bool = False,
) -> Dict[str, str]:
    ratio = score / max_score if max_score else 0.0
    thresholds = template.grade_thresholds or _load_templates()[0].grade_thresholds
    if critical:
        worst = thresholds[-1]
        return {
            "grade": worst.grade,
            "label": worst.label,
            "ratio": f"{ratio:.2f}",
            "note": "중대 위반 감지 → Zero-tolerance 적용",
        }
    for threshold in thresholds:
        if ratio >= threshold.min_ratio:
            return {
                "grade": threshold.grade,
                "label": threshold.label,
                "ratio": f"{ratio:.2f}",
                "note": "",
            }
    worst = thresholds[-1]
    return {
        "grade": worst.grade,
        "label": worst.label,
        "ratio": f"{ratio:.2f}",
        "note": "",
    }


def build_report(request: SupplierEvaluationRequest) -> str:
    template_section = generate_template_csv(request.supplier, request.industry)
    score_payload = score_supplier(request)
    risks = score_payload["risks"] or ["주요 리스크 특이사항 없음 (기본 수준)"]
    strengths = score_payload["strengths"] or ["강점 근거 부족"]
    benchmarks = score_payload["benchmarks"] or ["벤치마크 없음"]
    assets: List[str] = []
    if score_payload.get("xlsx_path"):
        assets.append(f"XLSX: {score_payload['xlsx_path']}")
    if score_payload.get("pdf_path"):
        assets.append(f"PDF: {score_payload['pdf_path']}")
    db_payload = json.dumps(score_payload.get("db_payload", {}), ensure_ascii=False)

    report_lines = [
        "[협력사 ESG 평가 템플릿]",
        template_section,
        "",
        "[Score Engine 결과]",
        score_payload["csv"],
        score_payload["summary"],
        *score_payload["global_notes"],
        "",
        "[등급 분류]",
        f"- 템플릿: {score_payload['template']} ({score_payload['version']})",
        f"- 등급: {score_payload['grade']} ({score_payload['grade_label']})",
        f"- 등급 메모: {score_payload.get('grade_note') or '해당 없음'}",
        f"- 총점/만점: {score_payload['total']:.1f} / {score_payload['max_score']:.1f}",
        f"- 비중 상위 리스크: {', '.join(risks[:3])}",
        f"- 강점 영역: {', '.join(strengths[:3])}",
        f"- 벤치마크 세트: {', '.join(benchmarks)}",
        "",
        "[평가 보고서 요약]",
        f"- 협력사명: {request.supplier}",
        f"- 업종: {request.industry}",
        f"- 분석 컨텍스트 샘플: {request.context.splitlines()[0][:80] if request.context else '컨텍스트 미제공'}",
        f"- 산출물 경로: {', '.join(assets) if assets else '생성 실패'}",
        "- 후속 권고: 시정조치 계획 제출 → 월간 추적 → PDF/DB 저장",
        "",
        "[DB Payload 미리보기]",
        db_payload,
    ]
    return "\n".join(report_lines)
