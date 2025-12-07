from __future__ import annotations

from dataclasses import dataclass

from .checklist import generate_checklist
from .iso31000 import run_iso31000_workflow
from .materiality import analyze_materiality
from .supplier_eval import SupplierEvaluationRequest, build_report, generate_template_csv
from .utils import extract_section, extract_tagged_value


@dataclass
class ParsedRequest:
    context: str
    question: str
    work_type: str | None
    supplier: str
    industry: str


class RiskToolOrchestrator:
    """Routes 사용자의 자연어 요청을 각 ESG 리스크 모듈로 전달한다."""

    def run(self, query: str, focus_area: str | None = None) -> str:
        parsed = self._parse_request(query)
        task = self._detect_task(query)
        if task == "checklist":
            return generate_checklist(parsed.work_type or focus_area)
        if task == "supplier_template":
            return generate_template_csv(parsed.supplier, parsed.industry)
        if task == "supplier_report":
            request = SupplierEvaluationRequest(parsed.supplier, parsed.industry, parsed.context)
            return build_report(request)
        if task == "materiality":
            return analyze_materiality(parsed.context, parsed.question)
        # Default to ISO 31000 기반 위험도 분석
        return run_iso31000_workflow(parsed.context, parsed.question)

    def _detect_task(self, query: str) -> str:
        lowered = query.lower()
        if any(keyword in lowered for keyword in ["체크리스트", "점검표", "inspection"]):
            return "checklist"
        if "협력" in lowered or "supplier" in lowered:
            if any(keyword in lowered for keyword in ["템플릿", "양식", "template"]):
                return "supplier_template"
            return "supplier_report"
        if any(keyword in lowered for keyword in ["trend", "경향", "중대성", "materiality"]):
            return "materiality"
        if any(keyword in lowered for keyword in ["iso", "위험도", "risk scoring", "risk analysis"]):
            return "iso"
        return "iso"

    def _parse_request(self, query: str) -> ParsedRequest:
        context = extract_section(query, "문서") or extract_section(query, "context") or query
        question = (
            extract_section(query, "분석 대상 질문")
            or extract_section(query, "질문")
            or extract_section(query, "query")
        )
        work_type = (
            extract_tagged_value(query, "작업유형")
            or extract_tagged_value(query, "작업 타입")
            or extract_tagged_value(query, "work_type")
        )
        supplier = extract_tagged_value(query, "협력사명") or "미지정 협력사"
        industry = extract_tagged_value(query, "업종") or "미지정 업종"
        return ParsedRequest(context=context, question=question, work_type=work_type, supplier=supplier, industry=industry)
