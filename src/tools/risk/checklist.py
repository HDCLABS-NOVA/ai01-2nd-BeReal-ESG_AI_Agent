from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document

from .utils import clamp, sentence_tokenize, to_csv

try:  # optional dependency during tests
    from retriever.retriever_pipeline import load_vectorstore, build_retriever, ESGRetriever
except Exception:  # pragma: no cover
    load_vectorstore = None
    build_retriever = None
    ESGRetriever = None

try:  # optional dependency
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None

#1. RAG(VectorDB)에서 Topic별로 문서를 불러와서 자동 생성
#2. 없다면 JSON 기반 외부 설정 불러오기 (_load_external_rows)
#3. 그래도 없다면 DEFAULT_ROWS 사용
#4. work_type이 지정되면 현장 특화 항목 추가

CHECKLIST_VERSION = "2025 Rev 4"
DATA_ROOT = Path("data/esg/checklists")
CSV_HEADERS = [
    "카테고리",
    "세부공종",
    "점검항목",
    "점검기준",
    "Hazard Code",
    "Hazard 분류",
    "Risk Factor",
    "Existing Control",
    "Additional Control",
    "Severity(1~5)",
    "Likelihood(1~5)",
    "Risk Rating",
    "법명",
    "조항",
    "세부내용",
    "예외조건",
    "점검결과",
    "비고",
    "근거문서",
]

HAZARD_CLASSIFICATION = {
    "FALL": "추락(Fall)",
    "STRUCK": "낙하물(Struck-by)",
    "CAUGHT": "협착(Caught-in/between)",
    "COLLAPSE": "붕괴(Collapse)",
    "ASPHYX": "질식(Asphyxiation)",
    "CHEM": "화학(Hazardous substance)",
    "ELEC": "전기(Electrical)",
    "FIRE": "화재(Fire) / 폭발",  # convenient grouping
    "ERGON": "근골격계(Ergonomic)",
}

HAZARD_BASE_SCORES = {
    "FALL": (5, 3),
    "STRUCK": (4, 3),
    "CAUGHT": (4, 3),
    "COLLAPSE": (5, 2),
    "ASPHYX": (5, 2),
    "CHEM": (4, 3),
    "ELEC": (5, 2),
    "FIRE": (5, 2),
    "ERGON": (3, 3),
}

HAZARD_ALIASES = {
    "추락": "FALL",
    "낙하": "STRUCK",
    "낙하물": "STRUCK",
    "협착": "CAUGHT",
    "붕괴": "COLLAPSE",
    "질식": "ASPHYX",
    "화학": "CHEM",
    "감전": "ELEC",
    "전기": "ELEC",
    "화재": "FIRE",
    "폭발": "FIRE",
    "근골격": "ERGON",
}

RESULT_OPTIONS = ["미점검", "적합", "부적합", "해당없음"]

CRITERION_PATTERNS: Sequence[re.Pattern] = [
    re.compile(pattern)
    for pattern in [
        r"(?:하여야|해야|필요)하다",
        r"기준을.*충족",
        r"설치(?:하여야|해야)",
        r"금지한다",
        r"허용(?:범위|기준)",
        r"유지하여야",
    ]
]

REGULATION_PATTERNS = [
    re.compile(
        r"(?P<law>[가-힣A-Za-z0-9·()\s]{2,50}?(?:법|규칙|가이드|Guide|ISO\s?\d{4,5}))"
        r"\s*(?P<article>제\d+조(?:의\d+)?(?:제\d+항)?|제\d+조의\d+|제\d+조 제\d+항|별표\s*\d+|별지\s*제?\d+호\s*서식)"
        r"(?:\s*(?P<suffix>에\s*따른다|에\s*의한다|기준|별표)?)"
    ),
    re.compile(r"(?P<article>별표\s*\d+|별지\s*제?\d+호\s*서식)")
]

HIGH_RISK_TRADES = {
    "전기",
    "해체",
    "굴착",
    "타워크레인",
    "양중",
    "밀폐공간",
    "고소작업",
    "용접",
}

NONCOMPLIANCE_WORDS = {"미실시", "미착용", "부적합", "위반", "없음", "미확인"}
PAST_INCIDENT_WORDS = {"재발", "반복", "이전 사고", "사망", "중대재해"}
AUTOMATION_WORDS = {"IoT", "모니터링", "자동", "센서"}

_VECTORSTORE = None
_RETRIEVER = None
_LLM = None


@dataclass
class RegulationRef: #법령
    law: str
    article: str = ""
    detail: str = ""
    exception: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {"법명": self.law, "조항": self.article, "세부내용": self.detail, "예외조건": self.exception}


@dataclass
class RiskProfile: #위험성평가 구조 (Severity × Likelihood)
    risk_factor: str
    existing_control: str
    additional_control: str
    severity: int = 3
    likelihood: int = 3

    @property
    def rating(self) -> int:
        return max(1, self.severity) * max(1, self.likelihood)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Risk Factor": self.risk_factor,
            "Existing Control": self.existing_control,
            "Additional Control": self.additional_control,
            "Severity": self.severity,
            "Likelihood": self.likelihood,
            "Risk Rating": self.rating,
        }


@dataclass
class ChecklistRow: #체크리스트 1행을 구성하는 최종 결과물
    category: str
    subcategory: str
    item: str
    criterion: str
    hazard_code: str
    regulation: RegulationRef
    risk_profile: RiskProfile
    result: str = field(default_factory=lambda: RESULT_OPTIONS[0])
    notes: str = ""
    photos_required: bool = False
    action_plan_hint: str = ""
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "category": self.category,
            "subcategory": self.subcategory,
            "item": self.item,
            "criterion": self.criterion,
            "hazard_code": self.hazard_code,
            "hazard_label": HAZARD_CLASSIFICATION.get(self.hazard_code, self.hazard_code),
            "regulation": self.regulation.to_dict(),
            "risk_profile": self.risk_profile.to_dict(),
            "result": self.result,
            "notes": self.notes,
            "photos_required": self.photos_required,
            "action_plan_hint": self.action_plan_hint,
            "source_metadata": self.source_metadata,
        }
        return payload

    def to_csv_row(self) -> List[str]:
        reg = self.regulation.to_dict()
        risk = self.risk_profile
        return [
            self.category,
            self.subcategory,
            self.item,
            self.criterion,
            self.hazard_code,
            HAZARD_CLASSIFICATION.get(self.hazard_code, self.hazard_code),
            risk.risk_factor,
            risk.existing_control,
            risk.additional_control,
            str(risk.severity),
            str(risk.likelihood),
            str(risk.rating),
            reg["법명"],
            reg["조항"],
            reg["세부내용"],
            reg["예외조건"],
            self.result,
            self.notes,
            self.source_metadata.get("source_file") if self.source_metadata else "",
        ]

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "ChecklistRow":
        hazard_code = _resolve_hazard(payload)
        regulation_data = payload.get("regulation", {})
        risk_data = payload.get("risk_profile", {})
        return ChecklistRow(
            category=payload.get("category", "기타"),
            subcategory=payload.get("subcategory", payload.get("sub_category", "일반")),
            item=payload.get("item") or payload.get("title", "Unnamed Item"),
            criterion=payload.get("criterion", payload.get("standard", "")),
            hazard_code=hazard_code,
            regulation=RegulationRef(
                regulation_data.get("law", regulation_data.get("법명", "")),
                regulation_data.get("article", regulation_data.get("조항", "")),
                regulation_data.get("detail", regulation_data.get("세부내용", "")),
                regulation_data.get("exception", regulation_data.get("예외조건", "")),
            ),
            risk_profile=RiskProfile(
                risk_data.get("risk_factor", risk_data.get("Risk Factor", "")),
                risk_data.get("existing_control", risk_data.get("Existing Control", "")),
                risk_data.get("additional_control", risk_data.get("Additional Control", "")),
                int(risk_data.get("severity", risk_data.get("Severity", 3) or 3)),
                int(risk_data.get("likelihood", risk_data.get("Likelihood", 3) or 3)),
            ),
            result=payload.get("result", RESULT_OPTIONS[0]),
            notes=payload.get("notes", ""),
            photos_required=payload.get("photos_required", False),
            action_plan_hint=payload.get("action_plan_hint", payload.get("Action Plan", "")),
            source_metadata=payload.get("source_metadata", {}),
        )


def _resolve_hazard(payload: Dict[str, Any]) -> str:
    code = payload.get("hazard_code") or payload.get("hazard", "").upper()
    if code in HAZARD_CLASSIFICATION:
        return code
    description = payload.get("hazard_label") or payload.get("hazard", "")
    for alias, mapped in HAZARD_ALIASES.items():
        if alias.lower() in description.lower():
            return mapped
    return "FALL"


DEFAULT_ROWS: List[Dict[str, Any]] = [
    {
        "category": "안전(Safety)",
        "subcategory": "가설/추락",
        "item": "작업발판 상태",
        "criterion": "발판 파손·침하·난간 설치 상태 점검",
        "hazard_code": "FALL",
        "regulation": {
            "law": "산업안전보건기준 규칙",
            "article": "제34조",
            "detail": "가설 통로·작업발판 구조기준",
        },
        "risk_profile": {
            "risk_factor": "난간 누락, 발판 이격",
            "existing_control": "일일 TBM·작업허가 확인",
            "additional_control": "추락방지망 및 안전대 추가 설치",
            "severity": 4,
            "likelihood": 3,
        },
        "photos_required": True,
        "action_plan_hint": "즉시 임시 작업 중지 후 보수",
    },
    {
        "category": "안전(Safety)",
        "subcategory": "전기/설비",
        "item": "전기설비 Lockout-Tagout",
        "criterion": "전원 차단·표지·검증 절차 이행",
        "hazard_code": "ELEC",
        "regulation": {
            "law": "KOSHA Guide C-31",
            "article": "LOTO 절차",
            "detail": "전기·기계 에너지 격리",
        },
        "risk_profile": {
            "risk_factor": "임시 분전함 개방, 연속 작업",
            "existing_control": "누전차단기·절연저항 점검",
            "additional_control": "이중 잠금, 감시자 지정",
            "severity": 5,
            "likelihood": 2,
        },
        "photos_required": False,
    },
    {
        "category": "환경(Environment)",
        "subcategory": "폐기물/화학",
        "item": "지정폐기물 분리·임시보관",
        "criterion": "라벨 부착·방수바닥·차폐",
        "hazard_code": "CHEM",
        "regulation": {
            "law": "폐기물관리법",
            "article": "제17조",
            "detail": "사업장 폐기물 보관기준",
        },
        "risk_profile": {
            "risk_factor": "혼합보관, 용량 초과",
            "existing_control": "보관대장·인계서 관리",
            "additional_control": "IoT 계량·CCTV",
            "severity": 4,
            "likelihood": 3,
        },
        "action_plan_hint": "위탁업체 교육 및 주간 점검",
    },
]

CHECKLIST_TOPICS: List[Dict[str, Any]] = [
    {
        "category": "안전(Safety)",
        "subcategory": "형틀/거푸집",
        "item": "작업발판·난간 설치",
        "hazard_code": "FALL",
        "query": "산업안전보건기준 규칙 제34조 작업발판 난간 설치 기준",
        "criterion_hint": "발판 파손·난간·개구부 방호 상태",
        "risk_factor_hint": "난간 미설치, 발판 침하",
        "existing_control_hint": "TBM·작업허가서 점검",
        "additional_control_hint": "추락방지망 추가 설치",
        "regulation_hint": {"law": "산업안전보건기준 규칙", "article": "제34조"},
    },
    {
        "category": "안전(Safety)",
        "subcategory": "전기/에너지 격리",
        "item": "전기설비 LOTO",
        "hazard_code": "ELEC",
        "query": "KOSHA Guide C-31 Lockout Tagout 전기 안전",
        "criterion_hint": "에너지 격리 절차, 표지, 검증",
        "risk_factor_hint": "임시 분전함 무단 개방",
        "existing_control_hint": "누전차단기·절연저항 측정",
        "additional_control_hint": "감시자 지정, 이중 잠금",
        "regulation_hint": {"law": "KOSHA Guide C-31"},
    },
    {
        "category": "안전(Safety)",
        "subcategory": "밀폐공간",
        "item": "밀폐공간 작업허가",
        "hazard_code": "ASPHYX",
        "query": "KOSHA C-81 밀폐공간 산소농도 작업허가",
        "criterion_hint": "산소·유해가스 측정, 환기, 감시자",
        "risk_factor_hint": "산소 결핍·황화수소",
        "existing_control_hint": "가스 측정·송풍기",
        "additional_control_hint": "웨어러블 가스센서",
        "regulation_hint": {"law": "KOSHA Guide C-81"},
    },
    {
        "category": "안전(Safety)",
        "subcategory": "굴착/토공",
        "item": "굴착·흙막이 안정성",
        "hazard_code": "COLLAPSE",
        "query": "KOSHA Guide H-96 굴착 흙막이 버팀 점검",
        "criterion_hint": "버팀·토류판·배수",
        "risk_factor_hint": "지반 연약, 수압",
        "existing_control_hint": "계측기 모니터링",
        "additional_control_hint": "IoT 변위계",
        "regulation_hint": {"law": "산업안전보건기준 규칙", "article": "제283조"},
    },
    {
        "category": "안전(Safety)",
        "subcategory": "양중/타워크레인",
        "item": "크레인 작업계획",
        "hazard_code": "STRUCK",
        "query": "산업안전보건기준 규칙 타워크레인 작업계획 기준",
        "criterion_hint": "인양하중, 신호수, 전도방지",
        "risk_factor_hint": "과부하, 풍속",
        "existing_control_hint": "하중계·LMI",
        "additional_control_hint": "지능형 제한장치",
        "regulation_hint": {"law": "산업안전보건기준 규칙", "article": "제79조"},
    },
    {
        "category": "안전(Safety)",
        "subcategory": "화기/용접",
        "item": "Hot Work Permit",
        "hazard_code": "FIRE",
        "query": "KOSHA Guide W-14 화기 작업허가",
        "criterion_hint": "불티 차단, 감시자, 소화기",
        "risk_factor_hint": "인화성 물질 근접",
        "existing_control_hint": "소화기 비치, 스파크 방호",
        "additional_control_hint": "열화상 모니터링",
        "regulation_hint": {"law": "KOSHA Guide W-14"},
    },
    {
        "category": "안전(Safety)",
        "subcategory": "중량물/협착",
        "item": "중량물 취급 및 끼임 방지",
        "hazard_code": "CAUGHT",
        "query": "KOSHA Guide M-4 중량물 취급 협착 방지",
        "criterion_hint": "와이어로프·훅 손상, 버클",
        "risk_factor_hint": "협착, 롤링",
        "existing_control_hint": "신호수 배치",
        "additional_control_hint": "근접센서",
        "regulation_hint": {"law": "KOSHA Guide M-4"},
    },
    {
        "category": "환경(Environment)",
        "subcategory": "폐기물",
        "item": "지정폐기물 보관/인계",
        "hazard_code": "CHEM",
        "query": "폐기물관리법 제17조 사업장 지정폐기물 보관 기준",
        "criterion_hint": "라벨·방수바닥·보관기간",
        "risk_factor_hint": "혼합보관, 누출",
        "existing_control_hint": "보관대장",
        "additional_control_hint": "IoT 무게센서",
        "regulation_hint": {"law": "폐기물관리법", "article": "제17조"},
    },
    {
        "category": "환경(Environment)",
        "subcategory": "비산먼지/소음",
        "item": "비산먼지·소음 관리",
        "hazard_code": "CHEM",
        "query": "대기환경보전법 비산먼지 저감 조치, 소음진동법 기준",
        "criterion_hint": "살수·방진막·측정 기록",
        "risk_factor_hint": "건조기후, 야간작업",
        "existing_control_hint": "살수차, 소음계",
        "additional_control_hint": "실시간 IoT 계측",
        "regulation_hint": {"law": "대기환경보전법", "article": "제43조"},
    },
    {
        "category": "환경(Environment)",
        "subcategory": "오염수/비점오염",
        "item": "오염수 차단",
        "hazard_code": "CHEM",
        "query": "물환경보전법 제28조 오염수 배출 관리",
        "criterion_hint": "배수로 차단, 침사지",
        "risk_factor_hint": "비산토, 침출수",
        "existing_control_hint": "침사지 운영",
        "additional_control_hint": "유량 모니터링",
        "regulation_hint": {"law": "물환경보전법", "article": "제28조"},
    },
    {
        "category": "환경(Environment)",
        "subcategory": "화학물질",
        "item": "화학물질 MSDS·누출",
        "hazard_code": "CHEM",
        "query": "화학물질관리법 제22조 MSDS 비치 기준",
        "criterion_hint": "MSDS, 이중용기, 차단시설",
        "risk_factor_hint": "MSDS 미비치, 누출",
        "existing_control_hint": "이중용기",
        "additional_control_hint": "흡착제 비치",
        "regulation_hint": {"law": "화학물질관리법", "article": "제22조"},
    },
    {
        "category": "환경(Environment)",
        "subcategory": "토양/생태",
        "item": "토양 및 생태 보호",
        "hazard_code": "CHEM",
        "query": "토양환경보전법 공사현장 토양오염 방지",
        "criterion_hint": "잔토·폐토 적정 처리",
        "risk_factor_hint": "오염토 반출",
        "existing_control_hint": "차수막",
        "additional_control_hint": "토양오염 측정",
        "regulation_hint": {"law": "토양환경보전법", "article": "제9조"},
    },
    {
        "category": "안전(Safety)",
        "subcategory": "철근/골조",
        "item": "철근 결속·거푸집 지지",
        "hazard_code": "FALL",
        "query": "국토안전관리원 철근 작업 안전 수칙 난간 거푸집",
        "criterion_hint": "철근 조립 중 추락방지 난간·작업발판 유지",
        "risk_factor_hint": "임시 난간 제거",
        "existing_control_hint": "수평안전대, 와이어 난간",
        "additional_control_hint": "이동식 작업대",
        "regulation_hint": {"law": "산업안전보건기준 규칙", "article": "제37조"},
    },
    {
        "category": "안전(Safety)",
        "subcategory": "전기",
        "item": "임시 전력·접지 관리",
        "hazard_code": "ELEC",
        "query": "산업안전보건기준 규칙 임시 전력 설비 접지",
        "criterion_hint": "임시 배선, 접지, 누전차단기",
        "risk_factor_hint": "임시분전함 개방, 접지 미실시",
        "existing_control_hint": "정기 절연저항 측정",
        "additional_control_hint": "스마트 누전 감시",
        "regulation_hint": {"law": "산업안전보건기준 규칙", "article": "제322조"},
    },
    {
        "category": "안전(Safety)",
        "subcategory": "기계설비",
        "item": "압축공기·배관 시험",
        "hazard_code": "FIRE",
        "query": "기계설비 압력시험 안전 기준 폭발",
        "criterion_hint": "압력시험 시 격리, 방호막, 압력계",
        "risk_factor_hint": "시험 중 파열",
        "existing_control_hint": "시험 전 밸브 점검",
        "additional_control_hint": "폭발방지 차폐",
        "regulation_hint": {"law": "산업안전보건기준 규칙", "article": "제329조"},
    },
    {
        "category": "안전(Safety)",
        "subcategory": "소방",
        "item": "소방배관·스프링클러 작업",
        "hazard_code": "FIRE",
        "query": "소방공사 안전 소방시설공사업법 작업기준",
        "criterion_hint": "용접 시 화기허가, 잔수 제거",
        "risk_factor_hint": "잔수·유증기 착화",
        "existing_control_hint": "드레인, 화재감시",
        "additional_control_hint": "가연성 물질 차단",
        "regulation_hint": {"law": "소방시설공사업법", "article": "시행규칙"},
    },
    {
        "category": "환경(Environment)",
        "subcategory": "조경/외부",
        "item": "토사·식생 보호",
        "hazard_code": "CHEM",
        "query": "조경 공사 토양 유실 방지 기준",
        "criterion_hint": "토사 유실 방지포, 보호구역 표시",
        "risk_factor_hint": "사면 노출",
        "existing_control_hint": "방수포, 휀스",
        "additional_control_hint": "비닐멀칭",
        "regulation_hint": {"law": "환경영향평가 협의내용"},
    },
    {
        "category": "안전(Safety)",
        "subcategory": "해체",
        "item": "해체 계획 및 낙하물 관리",
        "hazard_code": "STRUCK",
        "query": "건물 해체 작업 안전 해체계획 낙하물",
        "criterion_hint": "해체순서, 낙하물 방지통로",
        "risk_factor_hint": "동시해체, 지지 미확인",
        "existing_control_hint": "해체계획서 승인",
        "additional_control_hint": "비산 방지막",
        "regulation_hint": {"law": "산업안전보건법", "article": "제38조"},
    },
]

COMMON_TRADES = [
    "토공",
    "흙막이",
    "파일",
    "골조",
    "철근",
    "거푸집",
    "철골",
    "마감",
    "전기",
    "기계설비",
    "소방",
    "조경",
    "외부공사",
    "해체",
    "굴착",
    "포장",
]


def _load_external_rows() -> List[ChecklistRow]:
    rows: List[ChecklistRow] = []
    if not DATA_ROOT.exists():
        return rows
    for path in DATA_ROOT.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        payload_rows = data.get("rows") if isinstance(data, dict) else data
        if not payload_rows:
            continue
        for entry in payload_rows:
            try:
                rows.append(ChecklistRow.from_dict(entry))
            except Exception:
                continue
    return rows


def _get_vectorstore():
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE
    if load_vectorstore is None:
        return None
    try:
        _VECTORSTORE = load_vectorstore()
    except Exception:
        _VECTORSTORE = None
    return _VECTORSTORE


def _get_llm():
    global _LLM
    if _LLM is not None:
        return _LLM
    if ChatOpenAI is None:
        return None
    try:
        _LLM = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    except Exception:
        _LLM = None
    return _LLM


def _get_retriever():
    global _RETRIEVER
    if _RETRIEVER is not None:
        return _RETRIEVER
    if build_retriever is None:
        return None
    vectordb = _get_vectorstore()
    llm = _get_llm()
    if vectordb is None or llm is None:
        return None
    try:
        _RETRIEVER = build_retriever(llm, vectorstore=vectordb, top_k=6, fetch_k=40)
    except Exception:
        _RETRIEVER = None
    return _RETRIEVER


def _search_vectorstore(query: str, metadata_filter: Optional[Dict] = None) -> List[Document]:
    retriever = _get_retriever()
    if retriever is not None:
        try:
            return retriever.get_relevant_documents({"question": query, "metadata_filter": metadata_filter or {}})
        except Exception:
            pass
    store = _get_vectorstore()
    if store is None:
        return []
    try:
        return store.similarity_search(query, k=6)
    except Exception:
        return []


def _extract_regulation(text: str, hint: Dict[str, str] | None = None) -> RegulationRef:
    law = hint.get("law") if hint else ""
    article = hint.get("article", "") if hint else ""
    detail = hint.get("detail", "") if hint else ""
    exception = hint.get("exception", "") if hint else ""
    for pattern in REGULATION_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        if match.groupdict().get("law"):
            law = match.group("law").strip()
        if match.groupdict().get("article"):
            article = match.group("article").strip()
        span = match.span()
        detail = text[max(0, span[0] - 80) : min(len(text), span[1] + 120)].strip()
        suffix = match.groupdict().get("suffix")
        if suffix and suffix not in detail:
            detail = f"{detail} ({suffix.strip()})"
        break
    return RegulationRef(law or "미상", article, detail, exception)


def _infer_scores(hazard_code: str, text: str) -> tuple[int, int]:
    base = HAZARD_BASE_SCORES.get(hazard_code, (3, 3))
    severity, likelihood = base
    if any(keyword in text for keyword in ["중대", "사망", "폭발"]):
        severity += 1
    if any(keyword in text for keyword in ["반복", "재발", "빈번"]):
        likelihood += 1
    return clamp(severity, 1, 5), clamp(likelihood, 1, 5)


def _apply_risk_modifiers(severity: int, likelihood: int, text: str, topic: Dict[str, Any]) -> tuple[int, int]:
    trade = str(topic.get("subcategory", ""))
    text_lower = text.lower()
    if any(keyword in trade for keyword in HIGH_RISK_TRADES):
        severity += 1
    if any(word in text for word in NONCOMPLIANCE_WORDS):
        likelihood += 1
    if any(word in text for word in PAST_INCIDENT_WORDS):
        severity += 1
        likelihood += 1
    if any(word.lower() in text_lower for word in AUTOMATION_WORDS):
        likelihood -= 1
    severity = clamp(severity, 1, 5)
    likelihood = clamp(likelihood, 1, 5)
    return severity, likelihood


def _build_risk_profile(text: str, hazard_code: str, topic: Dict[str, Any]) -> RiskProfile:
    sentences = sentence_tokenize(text)
    def _pick_sentence(keywords: Sequence[str], default: str) -> str:
        for sentence in sentences:
            if any(keyword in sentence for keyword in keywords):
                return sentence.strip()
        return default

    risk_factor = topic.get("risk_factor_hint") or _pick_sentence(["위험", "사고", "위해"], sentences[0] if sentences else "위험요인 미상")
    existing = topic.get("existing_control_hint") or _pick_sentence(["점검", "관리", "조치", "설치"], sentences[1] if len(sentences) > 1 else "기존 통제 미상")
    additional = topic.get("additional_control_hint") or _pick_sentence(["추가", "강화", "개선"], sentences[2] if len(sentences) > 2 else "추가 통제 제안 필요")
    severity, likelihood = _infer_scores(hazard_code, text)
    severity, likelihood = _apply_risk_modifiers(severity, likelihood, text, topic)
    return RiskProfile(risk_factor, existing, additional, severity, likelihood)


def _select_criterion(text: str, fallback: str) -> str:
    sentences = sentence_tokenize(text)
    for pattern in CRITERION_PATTERNS:
        for sentence in sentences:
            cleaned = sentence.strip()
            if pattern.search(cleaned):
                return cleaned
    for sentence in sentences:
        cleaned = sentence.strip()
        if cleaned.endswith("다.") and len(cleaned) <= 200:
            return cleaned
    return fallback or (sentences[0].strip() if sentences else "")


def _build_row_from_topic(topic: Dict[str, Any], doc: Optional[Document]) -> ChecklistRow:
    text = doc.page_content.strip() if doc else ""
    hazard_code = topic.get("hazard_code") or _resolve_hazard({"hazard": text})
    criterion = _select_criterion(text, topic.get("criterion_hint", "")) if text else topic.get("criterion_hint", "")
    regulation_hint = topic.get("regulation_hint", {})
    if text:
        regulation = _extract_regulation(text, regulation_hint)
    else:
        regulation = RegulationRef(
            regulation_hint.get("law", "미상"),
            regulation_hint.get("article", ""),
            regulation_hint.get("detail", ""),
            regulation_hint.get("exception", ""),
        )
    risk_profile = _build_risk_profile(text, hazard_code, topic)
    source_note = ""
    metadata = {}
    if doc:
        metadata = doc.metadata or {}
        source_file = metadata.get("source_file") or metadata.get("source")
        page = metadata.get("page")
        source_note = f"{source_file or 'vector_db'} p.{page}" if page else (source_file or "vector_db")
    row = ChecklistRow(
        category=topic.get("category", "기타"),
        subcategory=topic.get("subcategory", "일반"),
        item=topic.get("item", "Unnamed Item"),
        criterion=criterion,
        hazard_code=hazard_code,
        regulation=regulation,
        risk_profile=risk_profile,
        notes=source_note,
        photos_required=topic.get("photos_required", False),
        action_plan_hint=topic.get("action_plan_hint", ""),
        source_metadata=metadata,
    )
    return row


def _rows_from_vectorstore() -> List[ChecklistRow]:
    rows: List[ChecklistRow] = []
    for topic in CHECKLIST_TOPICS:
        docs = _search_vectorstore(topic.get("query", ""), topic.get("metadata_filter"))
        doc = _choose_best_doc(docs, topic)
        rows.append(_build_row_from_topic(topic, doc))
    return [row for row in rows if row]


def _choose_best_doc(docs: Sequence[Document], topic: Dict[str, Any]) -> Optional[Document]:
    best = None
    best_score = -1
    hazard_code = topic.get("hazard_code", "")
    for doc in docs:
        text = doc.page_content
        score = 0
        for pattern in CRITERION_PATTERNS:
            if pattern.search(text):
                score += 3
        if hazard_code and hazard_code in text:
            score += 1
        if topic.get("subcategory") and topic["subcategory"] in text:
            score += 1
        if len(text) > 2000:
            score -= 1
        if score > best_score:
            best = doc
            best_score = score
    return best


def _build_worktype_rows(work_type: str) -> List[ChecklistRow]:
    template = {
        "category": "현장 특화",
        "subcategory": work_type,
        "item": f"{work_type} 작업허가 절차",
        "criterion": f"{work_type} 공종 위험성평가·TBM 확인",
        "hazard_code": "FALL",
        "regulation": {
            "law": "ISO 45001 절차서",
            "article": "작업허가",
        },
        "risk_profile": {
            "risk_factor": f"{work_type} 특화 위험",
            "existing_control": "작업허가서·감독자 승인",
            "additional_control": "스마트 태깅/IoT 감시",
            "severity": 4,
            "likelihood": 3,
        },
    }
    env_template = {
        "category": "현장 특화",
        "subcategory": work_type,
        "item": f"{work_type} 환경관리",
        "criterion": f"{work_type} 배출·소음 통제 계획",
        "hazard_code": "CHEM",
        "regulation": {
            "law": "ISO 14001 운영기준",
            "article": "환경관리계획",
        },
        "risk_profile": {
            "risk_factor": "오염수·비산먼지",
            "existing_control": "방진막·세륜시설",
            "additional_control": "실시간 IoT 계측",
            "severity": 3,
            "likelihood": 3,
        },
    }
    return [ChecklistRow.from_dict(template), ChecklistRow.from_dict(env_template)]


def generate_checklist(work_type: str | None = None) -> str:
    rows = _rows_from_vectorstore()
    source_candidates: List[str] = []
    if not rows:
        rows = _load_external_rows()
        source_candidates = [str(path.name) for path in DATA_ROOT.glob("*.json")]
    if not rows:
        rows = [ChecklistRow.from_dict(row) for row in DEFAULT_ROWS]
        source_candidates = ["DEFAULT_ROWS"]
    if work_type:
        rows.extend(_build_worktype_rows(work_type))
    vector_sources = {
        row.source_metadata.get("source_file")
        for row in rows
        if row.source_metadata and row.source_metadata.get("source_file")
    }
    sources = [src for src in vector_sources if src] or source_candidates
    json_payload = {
        "version": CHECKLIST_VERSION,
        "rows": [row.to_dict() for row in rows],
        "result_options": RESULT_OPTIONS,
        "source_files": sources,
        "recommended_output": ["xlsx", "모바일 앱", "PDF"],
    }
    csv_payload = to_csv(CSV_HEADERS, [row.to_csv_row() for row in rows])
    output = [
        "[ESG 현장 안전·환경 체크리스트(JSON)]",
        json.dumps(json_payload, ensure_ascii=False, indent=2),
        "",
        "[CSV Export (fallback)]",
        csv_payload,
        "",
        "※ Result/Action Plan/사진 첨부는 현장 앱 또는 엑셀에서 입력하십시오.",
    ]
    return "\n".join(output)
