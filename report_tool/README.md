# ESG Report Generator (GRI 2021)

GRI 2021 Standards를 완벽하게 준수하는 ESG 보고서 자동 생성 시스템

## 특징

✅ **GRI 2021 완전 준수** - 3단계 구조 (Universal/Sector/Topic)
✅ **자동 매핑** - 중대 이슈 → GRI Topic Standards
✅ **GRI Index 자동 생성** - 3단계 구조 테이블
✅ **다중 표준 통합** - K-ESG, ISO 26000, UN SDGs, SASB, TCFD
✅ **Core 라이브러리 분리** - `Tool` 패키지로 모듈화

## 파일 구조

```
esg-report-system/
├── esg_data.json             # 데이터 템플릿 (자동 로드용)
├── requirements.txt          # 의존성
├── README.md                 # 가이드
└── Tool/                     # 코어 라이브러리 (수정 불필요)
    ├── __init__.py
    ├── report_tool.py
    ├── esg_report_generator.py
    └── templates/
        └── report_template.html
```

## 빠른 시작

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. 사용 방법

프로젝트 루트에 실행 스크립트(예: `run.py`)를 만들고 아래와 같이 작성하세요.

```python
from Tool.report_tool import ReportTool

# 1. 도구 초기화
tool = ReportTool()

# 2. 데이터 수집
# - esg_data.json 자동 로드
# - 필수 데이터 누락 시 터미널에서 대화형 입력
tool.gather_data_interactive()

# 3. 추가 데이터 입력 (선택 사항)
# tool.store_data({"env_score": 95})

# 4. 보고서 생성
tool.create_report(report_path="esg_report.html")
```

### 3. 실행

```bash
python run.py
```

## 데이터 구조 (esg_data.json)

상위 폴더나 현재 폴더에 `esg_data.json`을 두면 자동으로 읽어옵니다.

### 필수 필드
- `company_name`: 회사명
- `report_year`: 보고 연도
- `ceo_message`: CEO 메시지
- `esg_strategy`: ESG 전략
- `env_policy`: 환경 정책
- `social_policy`: 사회 정책
- `gov_structure`: 지배구조
- `material_issues`: 중대 이슈 목록 (GRI 자동 매핑용)

## GRI 자동 매핑 프로세스

중대 이슈를 입력하면 자동으로 해당 GRI Topic Standards를 식별합니다.

```python
"기후변화 대응" → GRI 302 (에너지), GRI 305 (배출)
"안전보건" → GRI 403 (산업안전보건)
"공급망 관리" → GRI 308, GRI 414 (공급업체 평가)
"윤리경영" → GRI 205 (반부패)
```

## 보고서 구조 (HTML)

1. **Cover Page**: 보고서 표지
2. **CEO Message**: 경영진 메시지 (GRI 2-22)
3. **ESG Highlights**: 주요 성과 지표
4. **Materiality Assessment**: 중대성 평가 및 GRI 매핑 (GRI 3-1, 3-2)
5. **Environmental Performance**: 환경 성과 (GRI 300 Series)
6. **Social Performance**: 사회 성과 (GRI 400 Series)
7. **Governance & Ethics**: 지배구조 및 윤리 (GRI 200 Series)
8. **GRI Content Index**: GRI 2021 준수 인덱스 테이블

## 라이선스

MIT License
