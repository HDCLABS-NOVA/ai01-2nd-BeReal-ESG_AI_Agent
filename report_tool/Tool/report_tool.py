"""
Report Tool - ESG 보고서 데이터 통합 및 생성 인터페이스
------------------------------------------------------

정책·리스크 데이터 저장 및 Markdown/PDF 출력
"""

import shutil
import platform
import subprocess
import os
import json
from typing import Any, Dict, List, Optional
from .esg_report_generator import generate_esg_report

class DataLoader:
    """데이터 자동 로드 및 검색"""
    
    @staticmethod
    def find_and_load(filename: str = "esg_data.json") -> Dict[str, Any]:
        """상위 폴더에서 데이터 파일 검색 및 로드"""
        # 검색 경로: 현재 폴더 -> 상위 -> 상위의 상위
        search_paths = [
            os.path.join(".", filename),
            os.path.join("..", filename),
            os.path.join("..", "..", filename)
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        print(f"[Info] 데이터 파일 발견: {os.path.abspath(path)}")
                        return json.load(f)
                except Exception as e:
                    print(f"[Warning] 파일 로드 실패 ({path}): {e}")
        
        print("[Info] 자동 로드할 데이터 파일이 없습니다.")
        return {}

class ReportTool:
    """ESG 보고서 생성 도구"""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        
    def store_data(self, data: Dict[str, Any]) -> None:
        """데이터 저장 (기존 데이터와 병합)"""
        self._data.update(data)

    def load_from_file(self, filename: str = "esg_data.json"):
        """파일에서 데이터 자동 로드"""
        loaded = DataLoader.find_and_load(filename)
        if loaded:
            self.store_data(loaded)
            print(f" -> {len(loaded)}개 항목 로드됨")

    def gather_data_interactive(self):
        """대화형 데이터 입력 모드"""
        print("\n[Interactive Mode] 부족한 데이터를 입력받습니다.")
        
        # 1. 자동 로드 시도
        self.load_from_file()
        
        # 2. 필수 필드 확인 및 요청
        required_fields = {
            "company_name": "회사명",
            "report_year": "보고 연도",
            "ceo_message": "CEO 메시지",
            "esg_strategy": "ESG 전략",
            "env_policy": "환경 정책",
            "social_policy": "사회 정책",
            "gov_structure": "지배구조"
        }
        
        for key, label in required_fields.items():
            if key not in self._data:
                val = input(f" > {label} 입력: ").strip()
                if val:
                    self._data[key] = val
                    
        # 3. 중대 이슈 (복잡한 데이터는 안내만)
        if "material_issues" not in self._data:
            print("\n[!] 'material_issues'(중대 이슈) 데이터가 없습니다.")
            print("    이 데이터는 구조가 복잡하므로 코드나 JSON 파일로 입력을 권장합니다.")
            print("    예시: [{'name': '기후변화', 'impact': 80, 'isMaterial': True}]")
    
    def get_data(self) -> Dict[str, Any]:
        """저장된 데이터 반환"""
        return dict(self._data)
    
    def missing_fields(self) -> List[str]:
        """필수 필드 및 데이터 유효성 확인"""
        required = ["company_name", "report_year", "ceo_message", "esg_strategy", 
                   "env_policy", "social_policy", "gov_structure"]
        errors = [f"누락된 필드: {f}" for f in required if f not in self._data]
        
        # 데이터 유효성 검사
        if "material_issues" in self._data:
            for idx, issue in enumerate(self._data["material_issues"]):
                name = issue.get("name", f"Item {idx}")
                
                # Impact/Financial 점수 검사 (0-100)
                for field in ["impact", "financial"]:
                    val = issue.get(field)
                    if val is not None:
                        if not isinstance(val, (int, float)):
                            errors.append(f"이슈 '{name}': {field}는 숫자여야 합니다.")
                        elif not (0 <= val <= 100):
                            errors.append(f"이슈 '{name}': {field}는 0-100 사이여야 합니다 ({val}).")

        return errors
    
    def _get_pdf_tools(self) -> Dict[str, str]:
        """PDF 변환에 필요한 도구 경로 확인"""
        tools = {}
        
        # 1. Pandoc 확인
        pandoc = shutil.which("pandoc")
        if not pandoc:
            raise RuntimeError("Pandoc을 찾을 수 없습니다. 설치 후 PATH에 추가해주세요.\n"
                             "Windows: winget install JohnMacFarlane.Pandoc\n"
                             "Linux: sudo apt-get install pandoc")
        tools["pandoc"] = pandoc
        
        # 2. LibreOffice 확인
        # Windows: soffice, Linux: libreoffice/soffice
        libreoffice = shutil.which("libreoffice") or shutil.which("soffice")
        
        # Windows의 일반적인 설치 경로 확인 (PATH에 없을 경우)
        if not libreoffice and platform.system() == "Windows":
             common_paths = [
                 r"C:\Program Files\LibreOffice\program\soffice.exe",
                 r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"
             ]
             for path in common_paths:
                 if os.path.exists(path):
                     libreoffice = path
                     break
        
        if not libreoffice:
            raise RuntimeError("LibreOffice를 찾을 수 없습니다.\n"
                             "Windows: https://www.libreoffice.org/download/download-libreoffice/ 설치 필요\n"
                             "Linux: sudo apt-get install libreoffice")
        tools["libreoffice"] = libreoffice
        
        return tools

    def create_report(self, user_inputs: Dict[str, Any] = None, 
                     report_path: Optional[str] = None) -> str:
        """ESG 보고서 생성 (HTML)
        
        Parameters
        ----------
        user_inputs : dict, optional
            추가 데이터
        report_path : str, optional
            저장 경로 (.html 추천, .pdf도 지원)
        """
        # 데이터 병합
        data = self.get_data()
        if user_inputs:
            data.update(user_inputs)
            
        # 유효성 검사 경고
        validation_errors = self.missing_fields()
        if validation_errors:
            print("[Warning] 데이터 유효성 문제:")
            for err in validation_errors:
                print(f" - {err}")
        
        # HTML 보고서 생성
        report_html = generate_esg_report(data)
        
        # 파일 저장
        if report_path:
            ext = os.path.splitext(report_path)[1].lower()
            
            if ext == ".pdf":
                # HTML -> PDF 변환 (WeasyPrint 추천, 없으면 Chrome/Edge 헤드리스 사용)
                # 여기서는 간단히 기존 로직 유지하되, HTML을 저장하고 프린트 권장
                print("Notice: PDF 변환 기능은 HTML -> PDF 엔진이 필요합니다.")
                print("현재 버전에서는 HTML 파일 저장을 권장합니다.")
                
                # HTML 파일도 같이 저장
                html_path = report_path.replace(".pdf", ".html")
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(report_html)
                print(f"HTML 원본 저장: {html_path}")
                
            else:
                # HTML 저장
                if not report_path.endswith(".html"):
                    report_path += ".html"
                
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(report_html)
                print(f"HTML 리포트 생성 완료: {report_path}")
        
        return report_html
