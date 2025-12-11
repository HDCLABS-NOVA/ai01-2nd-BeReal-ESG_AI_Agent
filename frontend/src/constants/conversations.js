export const GUIDE_CONVERSATION_ID = "guide"
export const GUIDE_CONVERSATION = {
  id: GUIDE_CONVERSATION_ID,
  title: "ESG 웹 사용 가이드",
  isGuide: true,
}

export const GUIDE_MESSAGES = [
  {
    sender: "bot",
    text: "ESG Insight Agent에 오신 것을 환영합니다! ESG 규제 질의, 협력사 리스크 진단, 보고서 생성을 한 화면에서 확인할 수 있습니다.",
  },
  {
    sender: "user",
    text: "ESG 규제 동향 알려줘",
  },
  {
    sender: "bot",
    text: "📊 2025-12-08 기준 주요 동향\n- 환경부: 배출권거래제 4차 계획 수립 지침 입법예고 (탄소배출권 추가 할당 예정)\n- 국토부: 건설안전특별법 시행령 개정 초안 공개 (협력사 안전점검 의무 강화)\n- 고용노동부: 공급망 인권실사 로드맵 발표 (중대재해법 연계).",
  },
  {
    sender: "user",
    text: "체크리스트를 만들어줘",
  },
  {
    sender: "bot",
    text: "✅ K-ESG/ISO 31000 점검표 샘플\n1. 안전-가설/추락: 산업안전보건기준 규칙 제34조, 발판 파손·난간 설치 여부 기록, Risk Rating 12\n2. 안전-전기: KOSHA C-31 LOTO 절차 준수, 누전차단기·감시자 지정, Risk Rating 10\n3. 환경-폐기물: 폐기물관리법 제17조, 지정폐기물 라벨/방수바닥·IoT 계량 유지.",
  },
  {
    sender: "user",
    text: "원청 보고서를 참고해서 GAP 분석해줘",
  },
  {
    sender: "bot",
    text: "📈 2025 협력사 ESG 보고서 요약\n- 배출량: 원청 목표 45,000 tonCO₂eq 대비 자사 45,200 ton → 추가 200 ton 감축 필요\n- LTIR: 목표 0.20, 실적 0.15로 양호하나 협력사 교육 2곳 미이수\n- 공급망: 원청 KPI 대비 고위험 협력사 1곳 미개선 → 시정계획 제출 권고.",
  },
]

export const GUIDE_REPORTS = [
  {
    id: "guide-report-report",
    title: "2025 협력사 ESG 보고서 (샘플)",
    items: [
      "📊 요약: 배출량 52,000→45,200 tonCO₂eq, LTIR 0.54→0.15, 사외이사 3/5로 독립성 확보",
      "🔍 근거: RE100 로드맵, ISO 45001 안전 데이터, 공급망 고위험 1건 개선 중",
      "💡 권고: 탄소배출 추가 감축 계획 제출, 협력사 인권 교육 100% 달성, ISO 37001 심사 주기 유지",
    ],
  },
  {
    id: "guide-report-checklist",
    title: "ISO 31000 체크리스트 (샘플)",
    items: [
      "Safety-가설: 작업발판·난간 상태 / 산업안전보건기준 규칙 제34조 / Risk Rating 12",
      "Safety-전기: LOTO 절차, 누전차단기 / KOSHA C-31 / Risk Rating 10",
      "Environment-폐기물: 지정폐기물 라벨·방수바닥 / 폐기물관리법 제17조 / Risk Rating 12",
    ],
  },
]
