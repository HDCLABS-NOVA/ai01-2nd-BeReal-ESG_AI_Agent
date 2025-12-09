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
    "accuracy": "정확성", "balance": "균형", "clarity": "명확성",
    "comparability": "비교가능성", "completeness": "완전성",
    "sustainability_context": "지속가능성 맥락", "timeliness": "적시성",
    "verifiability": "검증가능성"
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
                md += f"#### {cat} ({series} Series)\n"
                md += "| GRI | 공시 | 지표 |\n|-----|------|------|\n"
                for code in codes:
                    info = GRI_TOPICS[code]
                    for num, title in info["indicators"].items():
                        md += f"| {code} | {num} | {title} |\n"
                md += "\n"
        
        return md


# ============================================================================
# 보고서 생성 (Jinja2)
# ============================================================================

from jinja2 import Environment, FileSystemLoader
import os

def _val(arr: List[Dict], year: str) -> str:
    """연도별 값 추출 (포맷팅 포함)"""
    for row in arr:
        if str(row.get("year", "")).startswith(year):
            val = row.get("value", "-")
            return f"{val:,}" if isinstance(val, (int, float)) else str(val)
    return "-"

def generate_esg_report(data: Dict[str, Any]) -> str:
    """Jinja2 템플릿 기반 ESG 보고서 생성"""
    
    # 1. 템플릿 로드
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report_template.html")
    
    # 2. 데이터 컨텍스트 준비
    context = _prepare_context(data)
    
    # 3. 렌더링
    return template.render(context)

def _prepare_context(data: Dict[str, Any]) -> Dict[str, Any]:
    """템플릿용 데이터 컨텍스트 구성"""
    year = data.get("report_year", "2025")
    company = data.get("company_name", "Company Name")
    
    # GRI 매핑 분석
    mapper = GRIMapper()
    issues = data.get("material_issues", [])
    mapper.analyze_issues(issues)
    
    # Highlights 데이터 가공
    env_pts = data.get("env_chart_data", [])
    safe_pts = data.get("safety_chart_data", [])
    
    # Material Issues 가공
    processed_issues = []
    for issue in issues:
        if issue.get("isMaterial"):
            name = issue.get("name", "")
            mapped = []
            for kw, codes in MATERIALITY_TO_GRI.items():
                if kw in name.lower(): mapped.extend(codes)
            
            processed_issues.append({
                "name": name,
                "impact": issue.get("impact", 0),
                "gri_codes": ", ".join(sorted(set(mapped))) if mapped else "-",
            })
            
    # Tags helper
    def get_tags(prefix=None, exact_list=None):
        tags = set()
        if exact_list:
            tags.update(exact_list)
        if prefix:
            for code in mapper.applicable_gri:
                if code.startswith(prefix): tags.add(code)
        # 3-3은 항상 포함 (관리 접근 방식)
        if "GRI 3-3" not in tags and prefix: 
             tags.add("GRI 3-3") 
        return sorted(list(tags))

    tags = {
        "env_climate": get_tags(exact_list=['GRI 302', 'GRI 305'] if 'GRI 302' in mapper.applicable_gri or 'GRI 305' in mapper.applicable_gri else []),
        "social_safety": get_tags(exact_list=['GRI 403']),
        "social_supply": get_tags(exact_list=['GRI 308', 'GRI 414']),
        "gov_struct": get_tags(exact_list=['GRI 2-9', 'GRI 2-10']),
        "gov_ethics": get_tags(exact_list=['GRI 205', 'GRI 2-23', 'GRI 2-26'])
    }
    
    # GRI Index Data Structure
    gri_index = mapper.get_index_data()

    return {
        "company_name": company,
        "report_year": year,
        "formatted_ceo_message": data.get("ceo_message", "").replace("\n", "<br>"),
        
        # Highlights
        "highlights": {
            "env": _val(env_pts, year),
            "social": _val(safe_pts, year)
        },
        
        # Sections
        "material_issues": processed_issues,
        "climate_action": data.get("climate_action", "-"),
        "env_policy": data.get("env_policy", "-"),
        "env_data": env_pts,
        
        "safety_management": data.get("safety_management", "-"),
        "supply_chain_policy": data.get("supply_chain_policy", "-"),
        "supply_chain_risk": data.get("supply_chain_risk", []),
        
        "gov_structure": data.get("gov_structure", "-"),
        "ethics": data.get("ethics", "-"),
        
        "tags": tags,
        "gri_index": gri_index
    }

# GRIMapper 클래스에 데이터 반환 메서드 추가 필요
# 기존 method(generate_index_html) 대신 get_index_data를 사용하도록 변경해야 함
def get_index_data(self) -> Dict[str, List[Dict]]:
    gri2_map = {
        "2-1": ("Company Overview", "5"), "2-22": ("CEO Message", "7"),
        "2-9": ("Governance", "65"), "2-29": ("Stakeholder", "15")
    }
    
    gri2 = []
    for code, (loc, pg) in gri2_map.items():
        gri2.append({
            "code": f"GRI {code}",
            "title": GRI_2_DISCLOSURES[code]["title"],
            "loc": loc,
            "page": pg
        })
        
    topics = []
    if self.applicable_gri:
        for code in sorted(self.applicable_gri):
            if code in GRI_TOPICS:
                topics.append({
                    "code": code,
                    "title": GRI_TOPICS[code]["topic"],
                    "loc": f"{GRI_TOPICS[code]['cat']} Section",
                    "page": "-"
                })
                
    return {"gri2": gri2, "topics": topics}

# GRIMapper 메서드 패치 (동적으로 추가하거나 클래스 내부에 구현해야 함)
GRIMapper.get_index_data = get_index_data


# ----------------------------------------------------------------------------
# 하위 호환성 및 실행 코드 (Main)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    report = generate_esg_report(SAMPLE)
    # print(report) # 디버깅용
    print(f"HTML 리포트 생성 완료: {len(report):,}자")

