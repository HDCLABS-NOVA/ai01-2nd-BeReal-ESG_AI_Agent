from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .utils import sentence_tokenize, to_csv

try:  # Optional semantic search
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RatingBand:
    min_score: int
    label: str
    description: str
    treatment: str
    acceptance: str


@dataclass(frozen=True)
class RiskHazard:
    risk_id: str
    area: str
    source: str
    event: str
    consequence: str
    keywords: Tuple[str, ...]
    synonyms: Tuple[str, ...]
    default_likelihood: float
    default_impact: float
    controls: Tuple[str, ...]
    treatments: Tuple[str, ...]
    kpi: Tuple[str, ...]
    min_similarity: float

    def query(self) -> str:
        tokens = list(self.keywords) + list(self.synonyms) + [self.event, self.source]
        return " ".join(token for token in tokens if token)


@dataclass
class RiskEvidence:
    sentence: str
    similarity: float
    negated: bool
    notes: str


@dataclass
class Observation:
    likelihood: float
    impact: float
    evidence: RiskEvidence


@dataclass
class RiskAssessmentEntry:
    hazard: RiskHazard
    evidences: List[RiskEvidence] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    likelihood: float = 0.0
    impact: float = 0.0
    score: float = 0.0
    rating: str = ""
    rating_description: str = ""
    treatment: str = ""
    acceptance: str = ""
    dynamic_kpis: List[str] = field(default_factory=list)

    def record_observation(self, likelihood: float, impact: float, evidence: RiskEvidence) -> None:
        self.evidences.append(evidence)
        self.observations.append(Observation(likelihood, impact, evidence))

    def finalize(self, acceptance_rules: Dict[str, Dict[str, str]]) -> None:
        if not self.observations:
            return
        weights: List[float] = []
        for obs in self.observations:
            weight = 1.0 + min(0.5, obs.evidence.similarity)
            if obs.evidence.negated:
                weight *= 0.7
            weights.append(weight)
        total_weight = sum(weights) or 1.0
        agg_likelihood = sum(obs.likelihood * w for obs, w in zip(self.observations, weights)) / total_weight
        agg_impact = sum(obs.impact * w for obs, w in zip(self.observations, weights)) / total_weight
        evidence_bonus = min(0.7, math.log1p(len(self.observations)) * 0.25)
        self.likelihood = round(max(1.0, min(5.0, agg_likelihood + evidence_bonus)), 2)
        self.impact = round(max(1.0, min(5.0, agg_impact)), 2)
        self.score = round(self.likelihood * self.impact, 1)
        rating_band = _classify(self.score)
        self.rating = rating_band.label
        self.rating_description = rating_band.description
        self.treatment = rating_band.treatment
        area_rules = acceptance_rules.get(self.hazard.area, {})
        self.acceptance = area_rules.get(self.rating, rating_band.acceptance)
        self.dynamic_kpis = self._build_dynamic_kpis()

    def _build_dynamic_kpis(self) -> List[str]:
        dynamic = list(self.hazard.kpi)
        dynamic.append(f"{self.hazard.event} 발생 건수 (월 {max(1, round(self.likelihood))}회 이하)")
        dynamic.append(f"{self.hazard.area} 위험도 {self.rating} 이상 건수 0건 유지")
        return dynamic


BASE_DIR = Path(__file__).resolve().parents[3]
TAXONOMY_PATH = BASE_DIR / "data" / "iso31000_taxonomy.json"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_SENTENCES = 300


def _read_json(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


@lru_cache(maxsize=1)
def _load_config() -> Dict[str, object]:
    if not TAXONOMY_PATH.exists():
        raise RuntimeError("ISO 31000 위험 분류 파일을 찾을 수 없습니다.")
    return _read_json(TAXONOMY_PATH)


@lru_cache(maxsize=1)
def _load_hazards() -> Tuple[RiskHazard, ...]:
    config = _load_config()
    hazards: List[RiskHazard] = []
    for item in config.get("risk_items", []):
        hazards.append(
            RiskHazard(
                risk_id=item.get("id", "UNKNOWN"),
                area=item.get("area", "기타"),
                source=item.get("risk_source", ""),
                event=item.get("event", ""),
                consequence=item.get("consequence", ""),
                keywords=tuple(item.get("keywords", [])),
                synonyms=tuple(item.get("synonyms", [])),
                default_likelihood=float(item.get("default_likelihood", 3.0)),
                default_impact=float(item.get("default_impact", 3.0)),
                controls=tuple(item.get("controls", [])),
                treatments=tuple(item.get("treatments", [])),
                kpi=tuple(item.get("kpi", [])),
                min_similarity=float(item.get("min_similarity", 0.3)),
            )
        )
    if not hazards:
        raise RuntimeError("위험 항목이 정의되지 않았습니다.")
    return tuple(hazards)


@lru_cache(maxsize=1)
def _rating_bands() -> Tuple[RatingBand, ...]:
    config = _load_config()
    bands = [
        RatingBand(
            min_score=int(item.get("min_score", 1)),
            label=item.get("label", "Low"),
            description=item.get("description", ""),
            treatment=item.get("treatment", ""),
            acceptance=item.get("acceptance", "허용"),
        )
        for item in config.get("rating_matrix", [])
    ]
    if not bands:
        raise RuntimeError("위험 등급 매트릭스가 필요합니다.")
    bands.sort(key=lambda band: band.min_score, reverse=True)
    return tuple(bands)


@lru_cache(maxsize=1)
def _acceptance_rules() -> Dict[str, Dict[str, str]]:
    config = _load_config()
    rules = {}
    for area, mapping in config.get("acceptance_rules", {}).items():
        rules[area] = {label: status for label, status in mapping.items()}
    return rules


@lru_cache(maxsize=1)
def _negation_tokens() -> Tuple[str, ...]:
    config = _load_config()
    return tuple(config.get("negation_tokens", []))


@lru_cache(maxsize=1)
def _likelihood_modifiers() -> Dict[str, Tuple[str, ...]]:
    config = _load_config()
    inc = tuple(config.get("likelihood_modifiers", {}).get("increase", []))
    dec = tuple(config.get("likelihood_modifiers", {}).get("decrease", []))
    return {"increase": inc, "decrease": dec}


@lru_cache(maxsize=1)
def _impact_modifiers() -> Dict[str, Tuple[str, ...]]:
    config = _load_config()
    inc = tuple(config.get("impact_modifiers", {}).get("increase", []))
    dec = tuple(config.get("impact_modifiers", {}).get("decrease", []))
    return {"increase": inc, "decrease": dec}


@lru_cache(maxsize=1)
def _embedding_model() -> SentenceTransformer | None:
    if SentenceTransformer is None:
        return None
    model_name = os.getenv("ISO31000_EMBED_MODEL", DEFAULT_MODEL)
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.error("ISO31000 임베딩 모델 로드 실패(%s): %s", model_name, exc)
        return None


class SemanticSearcher:
    def __init__(self, contexts: Sequence[Dict[str, object]]) -> None:
        self.contexts = list(contexts)
        self.texts = [ctx["text"] for ctx in self.contexts]
        self.embeddings = self._embed(self.texts)

    @staticmethod
    def _embed(sentences: Sequence[str]) -> np.ndarray | None:
        model = _embedding_model()
        if not model or not sentences:
            return None
        return model.encode(list(sentences), normalize_embeddings=True)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        if not query or not self.texts:
            return []
        if self.embeddings is None:
            return self._lexical(query, top_k)
        model = _embedding_model()
        if model is None:
            return self._lexical(query, top_k)
        query_vec = model.encode([query], normalize_embeddings=True)[0]
        sims = np.dot(self.embeddings, query_vec)
        top_indices = np.argsort(-sims)[:top_k]
        return [(self.contexts[idx], float(sims[idx])) for idx in top_indices]

    def _lexical(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        tokens = [token for token in query.lower().split() if len(token) >= 2]
        scores: List[Tuple[str, float]] = []
        for ctx in self.contexts:
            lowered = ctx["text"].lower()
            score = sum(lowered.count(token) for token in tokens)
            if score:
                scores.append((ctx, float(score)))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]


class ContextStore:
    def __init__(self, text: str) -> None:
        sentences = sentence_tokenize(text)
        trimmed = sentences[:MAX_SENTENCES]
        self.contexts: List[Dict[str, object]] = []
        for idx, sentence in enumerate(trimmed):
            self.contexts.append({"text": sentence, "kind": "sentence", "index": idx})
        window_size = 2
        for idx in range(len(trimmed) - window_size + 1):
            chunk = " ".join(trimmed[idx : idx + window_size])
            self.contexts.append({"text": chunk, "kind": "window", "index": idx})

    def items(self) -> Sequence[Dict[str, object]]:
        return self.contexts


def _is_negated(sentence: str, hazard: RiskHazard) -> bool:
    lowered = sentence.lower()
    tokens = _negation_tokens()
    if not tokens:
        return False
    keywords = hazard.keywords + hazard.synonyms
    if not any(keyword.lower() in lowered for keyword in keywords):
        return False
    return any(token in lowered for token in tokens)


def _adjust_score(base: float, modifiers: Dict[str, Tuple[str, ...]], sentence: str, direction: str) -> float:
    words = modifiers.get(direction, tuple())
    for word in words:
        if word and word in sentence:
            base += 0.5 if direction == "increase" else -0.5
    return base


def _score_sentence(hazard: RiskHazard, sentence: str, negated: bool) -> Tuple[float, float, str]:
    likelihood = hazard.default_likelihood
    impact = hazard.default_impact
    lowered = sentence.lower()
    likelihood = _adjust_score(likelihood, _likelihood_modifiers(), lowered, "increase")
    likelihood = _adjust_score(likelihood, _likelihood_modifiers(), lowered, "decrease")
    impact = _adjust_score(impact, _impact_modifiers(), lowered, "increase")
    impact = _adjust_score(impact, _impact_modifiers(), lowered, "decrease")
    note = ""
    if negated:
        likelihood -= 1.0
        impact -= 0.5
        note = "완화 표현 감지"
    likelihood = round(max(1.0, min(5.0, likelihood)), 1)
    impact = round(max(1.0, min(5.0, impact)), 1)
    return likelihood, impact, note


def _classify(score: float) -> RatingBand:
    for band in _rating_bands():
        if score >= band.min_score:
            return band
    return _rating_bands()[-1]


def identify_risks(context: str) -> List[RiskAssessmentEntry]:
    store = ContextStore(context)
    searcher = SemanticSearcher(store.items())
    hazards = _load_hazards()
    results: Dict[str, RiskAssessmentEntry] = {}

    for hazard in hazards:
        query = hazard.query()
        matches = searcher.search(query, top_k=4)
        for ctx, similarity in matches:
            sentence = ctx["text"]
            if similarity < hazard.min_similarity:
                continue
            negated = _is_negated(sentence, hazard)
            likelihood, impact, note = _score_sentence(hazard, sentence, negated)
            evidence = RiskEvidence(sentence=sentence.strip(), similarity=similarity, negated=negated, notes=note)
            entry = results.setdefault(hazard.risk_id, RiskAssessmentEntry(hazard=hazard))
            entry.record_observation(likelihood, impact, evidence)

    acceptance_rules = _acceptance_rules()
    for entry in results.values():
        entry.finalize(acceptance_rules)
    return sorted(results.values(), key=lambda entry: entry.score, reverse=True)


def _build_payload(entries: Sequence[RiskAssessmentEntry], question: str = "") -> Dict[str, object]:
    config = _load_config()
    distribution: Dict[str, int] = {}
    for entry in entries:
        distribution[entry.rating] = distribution.get(entry.rating, 0) + 1
    return {
        "version": config.get("version"),
        "question": question,
        "total_risks": len(entries),
        "distribution": distribution,
        "risks": [
            {
                "id": entry.hazard.risk_id,
                "area": entry.hazard.area,
                "source": entry.hazard.source,
                "event": entry.hazard.event,
                "consequence": entry.hazard.consequence,
                "likelihood": entry.likelihood,
                "impact": entry.impact,
                "score": entry.score,
                "rating": entry.rating,
                "rating_description": entry.rating_description,
                "acceptance": entry.acceptance,
                "treatment": entry.treatment,
                "controls": entry.hazard.controls,
                "treatments": entry.hazard.treatments,
                "kpi": entry.dynamic_kpis,
                "evidences": [
                    {
                        "sentence": evidence.sentence,
                        "similarity": round(evidence.similarity, 3),
                        "negated": evidence.negated,
                        "notes": evidence.notes,
                    }
                    for evidence in entry.evidences
                ],
            }
            for entry in entries
        ],
    }


def run_iso31000_workflow(context: str, question: str = "") -> str:
    if not context.strip():
        return "문서(context)가 제공되지 않아 ISO 31000 기반 분석을 수행할 수 없습니다."
    entries = identify_risks(context)
    if not entries:
        return "문서에서 의미 있는 위험 신호를 찾지 못했습니다. 구체적 근거를 제공해 주세요."
    payload = _build_payload(entries, question)
    csv_rows = []
    for entry in entries:
        evidence_text = " | ".join(evidence.sentence for evidence in entry.evidences[:2])
        csv_rows.append(
            [
                entry.hazard.area,
                f"{entry.hazard.event} ({entry.hazard.source})",
                evidence_text,
                f"{entry.likelihood:.1f}",
                f"{entry.impact:.1f}",
                f"{entry.score:.1f}",
                entry.rating,
                f"조치: {entry.treatment}",
            ]
        )
    csv_output = to_csv(
        ["영역", "위험 이벤트", "근거", "발생가능성", "영향도", "점수", "등급", "권고"],
        csv_rows,
    )
    header = "[ISO 31000 기반 위험도 분석 결과]"
    question_line = f"분석 질문: {question}" if question else ""
    payload_block = json.dumps(payload, ensure_ascii=False, indent=2)
    return "\n".join(filter(None, [header, question_line, csv_output, "", "[Risk Payload(JSON)]", payload_block]))
