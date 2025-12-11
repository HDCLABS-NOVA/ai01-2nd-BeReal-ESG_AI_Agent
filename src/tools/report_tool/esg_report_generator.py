"""
ESG Report Generator with GRI 2021 Standards
--------------------------------------------

GRI 2021 3단계 구조를 완벽하게 반영한 ESG 보고서 생성 시스템
- GRI 1 (Foundation): 보고 원칙
- GRI 2 (General Disclosures): 조직 정보  
- GRI 3 (Material Topics): 중대성 평가
- Topic Standards: 자동 매핑

통합 모듈: GRI 데이터베이스, 매핑 로직, 보고서 생성, 인덱스 자동 생성
"""

from typing import Dict, List, Any, Set, Optional


# ============================================================================
# GRI 2021 데이터베이스
# ============================================================================

GRI_1_PRINCIPLES = {
    "accuracy": "✅ 정확성", "balance": "⚖️ 균형", "clarity": "🔍 명확성",
    "comparability": "📊 비교가능성", "completeness": "📦 완전성",
    "sustainability_context": "🌍 지속가능성 맥락", "timeliness": "⏱️ 적시성",
    "verifiability": "🛡️ 검증가능성"
}

GRI_2_DISCLOSURES = {
    "2-1": {"title": "조직 세부 정보"}, "2-2": {"title": "지속가능성 보고 주체"},
    "2-3": {"title": "보고 기간·빈도·연락처"}, "2-6": {"title": "활동·가치사슬"},
    "2-7": {"title": "근로자"}, "2-9": {"title": "거버넌스 구조"},
    "2-10": {"title": "거버넌스 기구 임명"}, "2-12": {"title": "임팩트 관리 감독"},
    "2-14": {"title": "지속가능성 보고 역할"}, "2-22": {"title": "지속가능발전 전략"},
    "2-23": {"title": "정책 선언"}, "2-25": {"title": "부정적 임팩트 개선"},
    "2-26": {"title": "조언·우려 제기 메커니즘"}, "2-27": {"title": "법규 준수"},
    "2-29": {"title": "이해관계자 참여"}
}

GRI_3_REQUIREMENTS = {
    "3-1": "중대 주제 결정 프로세스", "3-2": "중대 주제 목록", "3-3": "중대 주제 관리"
}

# 중대 이슈 → GRI 자동 매핑
MATERIALITY_TO_GRI = {
    "기후변화": ["GRI 302", "GRI 305"], "탄소": ["GRI 305"], "에너지": ["GRI 302"],
    "안전": ["GRI 403"], "보건": ["GRI 403"],
    "공급망": ["GRI 308", "GRI 414"], "협력사": ["GRI 308", "GRI 414"],
    "윤리": ["GRI 205", "GRI 206"], "부패": ["GRI 205"],
    "인권": ["GRI 406", "GRI 407", "GRI 408", "GRI 409"],
    "물": ["GRI 303"], "수자원": ["GRI 303"], "생물다양성": ["GRI 304"],
    "폐기물": ["GRI 306"], "순환": ["GRI 301", "GRI 306"],
    "경제": ["GRI 201"], "재무": ["GRI 201"],
    "고용": ["GRI 401"], "인재": ["GRI 401", "GRI 404"], "교육": ["GRI 404"],
    "다양성": ["GRI 405"], "차별": ["GRI 406"],
    "지역": ["GRI 413"], "품질": ["GRI 416"], "정보": ["GRI 418"]
}

# GRI Topic Standards
GRI_TOPICS = {
    "GRI 201": {"topic": "경제 성과", "cat": "경제", "indicators": {"201-1": "경제가치 창출", "201-2": "기후변화 재무영향"}},
    "GRI 205": {"topic": "반부패", "cat": "경제", "indicators": {"205-1": "부패 위험", "205-2": "반부패 정책", "205-3": "부패 사건"}},
    "GRI 206": {"topic": "경쟁저해", "cat": "경제", "indicators": {"206-1": "경쟁저해행위"}},
    "GRI 301": {"topic": "원재료", "cat": "환경", "indicators": {"301-1": "원재료 사용", "301-2": "재생 원재료"}},
    "GRI 302": {"topic": "에너지", "cat": "환경", "indicators": {"302-1": "에너지 소비", "302-3": "에너지 집약도", "302-4": "에너지 감축"}},
    "GRI 303": {"topic": "물", "cat": "환경", "indicators": {"303-1": "물 상호작용", "303-3": "취수", "303-5": "물 소비"}},
    "GRI 304": {"topic": "생물다양성", "cat": "환경", "indicators": {"304-1": "생물다양성 서식지", "304-2": "생물다양성 영향"}},
    "GRI 305": {"topic": "배출", "cat": "환경", "indicators": {"305-1": "Scope 1", "305-2": "Scope 2", "305-3": "Scope 3", "305-4": "배출 집약도", "305-5": "배출 감축"}},
    "GRI 306": {"topic": "폐기물", "cat": "환경", "indicators": {"306-1": "폐기물 발생", "306-3": "발생한 폐기물"}},
    "GRI 308": {"topic": "공급업체 환경", "cat": "환경", "indicators": {"308-1": "환경 심사 공급업체", "308-2": "공급망 환경영향"}},
    "GRI 401": {"topic": "고용", "cat": "사회", "indicators": {"401-1": "신규채용·이직", "401-3": "육아휴직"}},
    "GRI 403": {"topic": "안전보건", "cat": "사회", "indicators": {"403-1": "안전보건 시스템", "403-2": "위험 식별", "403-9": "업무 상해"}},
    "GRI 404": {"topic": "교육", "cat": "사회", "indicators": {"404-1": "평균 훈련시간", "404-2": "역량 강화"}},
    "GRI 405": {"topic": "다양성", "cat": "사회", "indicators": {"405-1": "거버넌스 구성", "405-2": "기본급 비율"}},
    "GRI 406": {"topic": "차별금지", "cat": "사회", "indicators": {"406-1": "차별 사건"}},
    "GRI 407": {"topic": "결사의 자유", "cat": "사회", "indicators": {"407-1": "결사 침해 위험"}},
    "GRI 408": {"topic": "아동노동", "cat": "사회", "indicators": {"408-1": "아동노동 위험"}},
    "GRI 409": {"topic": "강제노동", "cat": "사회", "indicators": {"409-1": "강제노동 위험"}},
    "GRI 413": {"topic": "지역사회", "cat": "사회", "indicators": {"413-1": "지역사회 참여"}},
    "GRI 414": {"topic": "공급업체 사회", "cat": "사회", "indicators": {"414-1": "사회 심사 공급업체", "414-2": "공급망 사회영향"}},
    "GRI 416": {"topic": "고객 안전", "cat": "사회", "indicators": {"416-1": "제품 안전 평가"}},
    "GRI 418": {"topic": "개인정보", "cat": "사회", "indicators": {"418-1": "개인정보 위반"}}
}


class GRIMapper:
    """GRI 자동 매핑 및 인덱스 생성"""
    
    def __init__(self):
        self.applicable_gri: Set[str] = set()
    
    def analyze_issues(self, issues: List[Dict]) -> None:
        """중대 이슈 분석 및 GRI 매핑"""
        for issue in issues:
            if not issue.get("isMaterial"):
                continue
            name = issue.get("name", "").lower()
            for keyword, gri_codes in MATERIALITY_TO_GRI.items():
                if keyword in name:
                    self.applicable_gri.update(gri_codes)
    
    def generate_index(self) -> str:
        """GRI Contents Index 생성"""
        md = "## GRI Contents Index\n\n본 보고서는 GRI Standards 2021 준수\n\n"
        
        # GRI 1
        md += "### GRI 1: Foundation 2021\n"
        md += "**적용 원칙:** " + ", ".join(GRI_1_PRINCIPLES.values()) + "\n\n"
        
        # GRI 2
        md += "### GRI 2: General Disclosures 2021\n"
        md += "| 공시 | 제목 | 위치 | 페이지 |\n|-----|------|------|-------|\n"
        gri2_map = {
            "2-1": ("Company Overview", "5"), "2-2": ("About Report", "2"), "2-3": ("About Report", "2"),
            "2-6": ("Supply Chain", "45"), "2-7": ("Talent", "40"), "2-9": ("Governance", "65"),
            "2-10": ("Board", "69"), "2-12": ("Stakeholder", "15"), "2-14": ("Stakeholder", "15"),
            "2-22": ("CEO Message", "7"), "2-23": ("Ethics", "70"), "2-25": ("Supply CAP", "56"),
            "2-26": ("Ethics", "71"), "2-27": ("Ethics", "72"), "2-29": ("Stakeholder", "15")
        }
        for num in sorted(gri2_map.keys()):
            title = GRI_2_DISCLOSURES[num]["title"]
            loc, pg = gri2_map[num]
            md += f"| {num} | {title} | {loc} | {pg} |\n"
        md += "\n"
        
        # GRI 3
        md += "### GRI 3: Material Topics 2021\n"
        md += "| 공시 | 제목 | 위치 |\n|-----|------|------|\n"
        md += "| 3-1 | 중대 주제 결정 | Materiality Assessment |\n"
        md += "| 3-2 | 중대 주제 목록 | Material Issues Table |\n"
        md += "| 3-3 | 중대 주제 관리 | E/S/G 섹션 |\n\n"
        
        # Sector
        md += "### Sector Standards\n건설업 미발행 → SASB 대체\n\n"
        
        # Topics
        if self.applicable_gri:
            md += "### Topic Standards\n\n"
            cats = {"경제": [], "환경": [], "사회": []}
            for code in sorted(self.applicable_gri):
                if code in GRI_TOPICS:
                    cats[GRI_TOPICS[code]["cat"]].append(code)
            
            for cat, codes in cats.items():
                if not codes:
                    continue
                series = "200" if cat == "경제" else ("300" if cat == "환경" else "400")
                md += f"#### 🔹 {cat} ({series} Series)\n\n"
                md += "| GRI | 공시 | 지표 |\n|-----|------|------|\n"
                for code in codes:
                    info = GRI_TOPICS[code]
                    for num, title in info["indicators"].items():
                        md += f"| {code} | {num} | {title} |\n"
                md += "\n"
        
        return md


# ============================================================================
# 보고서 생성
# ============================================================================

def _val(arr: List[Dict], year: str) -> str:
    """연도별 값 추출"""
    for row in arr:
        if str(row.get("year", "")).startswith(year):
            return str(row.get("value", "-"))
    return "-"


def _tag(tags: List[str]) -> str:
    """GRI 태그 포맷팅"""
    return f"**[{', '.join(sorted(set(tags)))}]**" if tags else ""


def generate_esg_report(data: Dict[str, Any], standard: str = "GRI") -> str:
    """ESG 보고서 생성 (조건부 렌더링 적용)"""
    
    def has_data(val: Any) -> bool:
        if not val: return False
        if isinstance(val, str) and "입력" in val: return False
        if isinstance(val, list) and not val: return False
        return True

    # 데이터 추출
    company = data.get("company_name", "회사명")
    year = data.get("report_year", "연도")
    industry = data.get("industry", "Construction")
    ceo = data.get("ceo_message", "")
    strategy = data.get("esg_strategy", "")
    
    env_pol = data.get("env_policy", "")
    climate = data.get("climate_action", "")
    env_data = data.get("env_chart_data", [])
    
    social_pol = data.get("social_policy", "")
    safety = data.get("safety_management", "")
    safety_data = data.get("safety_chart_data", [])
    supply_pol = data.get("supply_chain_policy", "")
    supply_risk = data.get("supply_chain_risk", [])
    
    gov = data.get("gov_structure", "")
    ethics = data.get("ethics", "")
    
    # ---------------------------------------------------------
    # 보고서 시작
    # ---------------------------------------------------------
    title_suffix = "지속가능경영보고서" if standard == "GRI" else "K-ESG 가이드라인 보고서"
    md = f"# {company} {year} {title_suffix}\n\n"
    
    # About (Always show)
    gri_tag = "**[GRI 2-1, 2-2, 2-3]**\n" if standard == "GRI" else ""
    md += f"## 📘 About This Report\n\n{gri_tag}\n"
    md += f"- **📅 기간:** {year}.1.1 ~ {year}.12.31\n"
    md += f"- **🏢 범위:** {company} 본사, 자회사, 1~2차 협력사\n"
    
    if standard == "GRI":
        md += "**기준:** GRI 2021, K-ESG, ISO 26000, UN SDGs, SASB, TCFD, CSRD\n"
    else:
        md += "**기준:** K-ESG 가이드라인 v2.0\n"
    md += "\n"
    
    # Highlights (show if data exists)
    if env_data or safety_data:
        md += "## 🏆 ESG Highlights\n\n"
        md += f"| 분야 | 2023 | 2024 | {year} |\n|------|------|------|------|\n"
        md += f"| 🌿 환경(GHG) | {_val(env_data,'2023')} | {_val(env_data,'2024')} | {_val(env_data,'2025')} |\n"
        md += f"| 👷 사회(LTIR) | {_val(safety_data,'2023')} | {_val(safety_data,'2024')} | {_val(safety_data,'2025')} |\n"
        md += "| 🏛️ 지배구조 | - | - | - |\n\n"
    
    # CEO Message (Removed as per user request)
    # if has_data(ceo):
    #     tag = "**[GRI 2-22]**\n\n" if standard == "GRI" else ""
    #     md += f"## CEO Message\n{tag}{ceo}\n\n"
    
    # Company Overview
    md += "## 🏢 Company Overview\n\n"
    md += f"- **회사명:** {company}\n- **업종:** {industry}\n\n"
    if has_data(strategy):
        md += f"### 🚀 전략\n\n{strategy}\n\n"
    
    # Stakeholder
    # Only show generic stakeholder table if it's a standard report (no custom sections)
    custom_sections = data.get("custom_sections", [])
    
    if not custom_sections:
        md += "## 🤝 ESG & Stakeholder Engagement\n\n"
        md += "이해관계자 소통 채널 운영 현황\n\n"
        md += "| 이해관계자 | 관심사 | 채널 |\n|------------|--------|------|\n"
        md += "| 👥 고객 | 안전·품질 | VOC |\n| 👷 임직원 | 안전·교육 | 교육 |\n"
        md += "| 🏗️ 협력사 | ESG | 포털 |\n| 💰 투자자 | 공시 | IR |\n| 🏙️ 지역사회 | 환경 | 봉사 |\n\n"
    
    # Materiality (Only if issues exist)
    mapper = GRIMapper()
    issues = data.get("material_issues", [])
    if issues:
        mapper.analyze_issues(issues) # Run mapping
    
    # Render Materiality
    md += "## 📌 Double Materiality Assessment\n\n"
    md += f"### 주요 이슈 도출 ({len(issues)}건)\n\n"
    md += "| 이슈 | 중요도(%) | 재무영향(%) | 관련 영역 |\n|------|---------|---------|-----|\n"
    for issue in issues:
        ref_str = "-"
        name_lower = issue.get("name", "").lower()
        
        # Simple E/S/G inference for K-ESG
        categories = []
        for kw, codes in MATERIALITY_TO_GRI.items():
            if kw in name_lower:
                # GRI_TOPICS has 'cat' field (e.g. '환경', '사회', '경제')
                # Take first code to find category
                first_code = codes[0]
                if first_code in GRI_TOPICS:
                    categories.append(GRI_TOPICS[first_code]['cat'])
        
        if categories:
             # Unique sorted categories (e.g. "환경, 사회")
             ref_str = ", ".join(sorted(set(categories)))
        else:
             ref_str = "일반"

        md += f"| {issue['name']} | {issue['impact']} | {issue['financial']} | {ref_str} |\n"
    md += "\n"
    
    # ---------------------------------------------------------
    # Custom / Dynamic Sections (Proposed Flexibility)
    # ---------------------------------------------------------
    if custom_sections:
        for section in custom_sections:
            title = section.get("title", "Section")
            content = section.get("content", "")
            md += f"## 🚩 {title}\n\n{content}\n\n"
            
    # Standard Sections (Environmental, Social, Governance)
    # These will naturally be skipped if the LLM left them empty as instructed.
    
    # Environmental
    if has_data(env_pol) or has_data(climate) or env_data:
        md += "## 🌿 Environmental Performance\n\n"
        if has_data(env_pol): md += f"### 📜 Policy\n\n{env_pol}\n\n"
        if has_data(climate): md += f"### 🌍 Climate Action\n\n{climate}\n\n"
        if env_data:
            md += "### 📉 Key Indicators\n\n"
            for r in env_data:
                md += f"- {r.get('year')}: {r.get('value')}\n"
            md += "\n"

    # Social
    if has_data(social_pol) or has_data(safety) or safety_data or has_data(supply_pol):
        md += "## 👥 Social Performance\n\n"
        if has_data(social_pol): md += f"### ⚖️ Human Rights\n\n{social_pol}\n\n"
        if has_data(safety): md += f"### 🦺 Safety Management\n\n{safety}\n\n"
        if safety_data:
            md += "#### 📊 Safety KPIs\n\n"
            for r in safety_data:
                md += f"- {r.get('year')}: {r.get('value')}\n"
            md += "\n"
        if has_data(supply_pol):
            md += f"### 🏗️ Supply Chain\n\n{supply_pol}\n\n"
            if supply_risk:
                md += "| 카테고리 | 리스크 | 조치 | 현황 |\n|----------|--------|------|------|\n"
                for r in supply_risk:
                    md += f"| {r.get('category')} | {r.get('riskLevel')} | {r.get('action')} | {r.get('status')} |\n"
                md += "\n"

    # Governance
    if has_data(gov) or has_data(ethics):
        md += "## ⚖️ Governance\n\n"
        if has_data(gov): md += f"### 🏛️ Structure\n\n{gov}\n\n"
        # Committees table removed as it was hardcoded.
        if has_data(ethics): md += f"### 📜 Ethics\n\n{ethics}\n\n"

    # Appendices
    if data.get("esg_data_details") or standard == "GRI":
        md += "---\n# Appendices\n\n"
    
    # B: ESG Data (Only if details exist)
    if data.get("esg_data_details"):
        md += "## ESG Data Details\n"
        for s in data["esg_data_details"]:
            md += f"### {s.get('title')}\n{s.get('content')}\n\n"
    
    # C: Index
    if standard == "GRI":
        md += "## GRI Content Index\n"
        md += mapper.generate_index()
    else:
        # K-ESG Index - only show if we have content for it, otherwise skipping as requested
        # Currently no data for it, so omitting to avoid "Empty Section" complaints.
        pass
    
    return md


# 샘플 데이터
SAMPLE = {
    "company_name": "(주)코리아건설",
    "report_year": "2025",
    "ceo_message": "지속가능경영 실천\n\n대표이사 김철수",
    "esg_strategy": "친환경·상생·거버넌스",
    "env_policy": "탄소중립 2050",
    "climate_action": "RE100, 감축",
    "env_chart_data": [{"year":"2023","value":52000},{"year":"2024","value":48500},{"year":"2025","value":45200}],
    "social_policy": "UN·ILO 준수",
    "safety_management": "ISO 45001",
    "safety_chart_data": [{"year":"2023","value":0.54},{"year":"2024","value":0.32},{"year":"2025","value":0.15}],
    "supply_chain_policy": "행동강령",
    "supply_chain_risk": [{"category":"안전","riskLevel":"High","action":"실사","status":"진행중"}],
    "gov_structure": "사내2·사외3",
    "ethics": "ISO 37001",
    "sasb_index": True,
    "material_issues": [
        {"name":"기후변화 대응","impact":85,"financial":90,"isMaterial":True},
        {"name":"안전보건","impact":95,"financial":88,"isMaterial":True},
        {"name":"공급망 관리","impact":80,"financial":75,"isMaterial":True}
    ]
}


if __name__ == "__main__":
    report = generate_esg_report(SAMPLE)
    print(f"보고서 생성: {len(report):,}자")
