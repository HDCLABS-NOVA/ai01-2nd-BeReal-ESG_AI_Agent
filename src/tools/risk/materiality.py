from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from .iso31000 import RiskEntry, identify_risks
from .utils import to_csv


SUPPLY_CHAIN_IMPACT = {
    "안전": "장비 임대·하도급 작업자의 안전 성숙도에 좌우",
    "환경": "폐기물·배출 처리 협력사 관리 미흡 시 공사 중단",  # supply chain view
    "노동": "인력 파견사 근로조건 불일치 위험",
    "거버넌스": "입찰·조달 과정 부패 리스크",
}

STAKEHOLDER_IMPACT = {
    "안전": "근로자·감리·발주처 불안감 증폭",
    "환경": "지역사회·주민 민원 증가",
    "노동": "근로자 이탈·노동청 민원",
    "거버넌스": "투자자·규제기관 신뢰 하락",
}

SYSTEMIC_IMPACT = {
    "안전": "산업 전반 안전문화 저하",
    "환경": "지역 생태계 및 기후목표 영향",
    "노동": "건설업 전반 인력수급 불안",
    "거버넌스": "공정경쟁 체계 훼손",
}


def _trend_summary(risks: List[RiskEntry], context: str) -> Tuple[str, str, str]:
    lowered = context.lower()
    if any(word in lowered for word in ["증가", "악화", "빈번"]):
        summary = "증가"
    elif any(word in lowered for word in ["감소", "개선", "완화"]):
        summary = "감소"
    else:
        summary = "정체"
    drivers = []
    if "법" in lowered or "규제" in lowered:
        drivers.append("법규 변화")
    if any(entry.rating == "High" for entry in risks):
        drivers.append("High risk 빈도")
    if "협력" in lowered or "공급" in lowered:
        drivers.append("공급망 영향")
    if not drivers:
        drivers.append("문서 기반 일반 추정")
    evidence = risks[0].evidence if risks else "문서에서 명시된 근거 문장 필요"
    return summary, ", ".join(drivers), evidence


def _materiality_level(entry: RiskEntry) -> Tuple[str, str]:
    impact_level = "High" if entry.rating == "High" else "Medium" if entry.rating == "Medium" else "Low"
    if entry.score >= 16:
        financial = "High"
    elif entry.score >= 9:
        financial = "Medium"
    else:
        financial = "Low"
    return impact_level, financial


def analyze_materiality(context: str, question: str = "") -> str:
    if not context.strip():
        return "Trend/Materiality 분석을 위해 문서(context)가 필요합니다."
    risks = identify_risks(context)
    if not risks:
        return "문서에서 리스크 근거를 찾지 못했습니다. 추가 데이터를 제공해 주세요."
    summary, drivers, evidence = _trend_summary(risks, context)
    double_rows = []
    for entry in risks:
        impact_level, financial_level = _materiality_level(entry)
        double_rows.append((entry.area, entry.hazard, impact_level, financial_level, entry.evidence))
    double_csv = to_csv(["영역", "리스크요인", "Impact Materiality", "Financial Materiality", "근거"], double_rows)
    triple_rows = []
    for entry in risks:
        triple_rows.append(
            (
                entry.area,
                entry.hazard,
                SUPPLY_CHAIN_IMPACT.get(entry.area, ""),
                STAKEHOLDER_IMPACT.get(entry.area, ""),
                SYSTEMIC_IMPACT.get(entry.area, ""),
            )
        )
    triple_csv = to_csv(["영역", "리스크요인", "Supply Chain 영향", "Stakeholder 영향", "Systemic 영향"], triple_rows)
    top_risks = sorted(risks, key=lambda r: r.score, reverse=True)[:5]
    top_lines = [f"- {entry.area}/{entry.hazard}: {entry.rating} (Score {entry.score})" for entry in top_risks]
    action_lines = [
        "1) ISO 31000 기반 정기 리스크 검토",
        "2) 공급망 협력사 ESG 실사 강화",
        "3) KPI 연동 모니터링 대시보드 구축",
    ]
    output_sections = [
        f"Trend Summary: {summary}",
        f"Trend Drivers: {drivers}",
        f"근거 문장: {evidence}",
        "",
        "[Double Materiality]",
        double_csv,
        "",
        "[Triple Materiality]",
        triple_csv,
        "",
        "고위험 리스크 요약",
        "\n".join(top_lines) if top_lines else "- 고위험 항목 없음",
        "",
        "중요성 평가 결과 요약",
        "- Impact/Financial Materiality를 종합하여 중점 관리 리스크를 도출",
        "",
        "Action Plan",
        "\n".join(action_lines),
    ]
    return "\n".join(output_sections)
