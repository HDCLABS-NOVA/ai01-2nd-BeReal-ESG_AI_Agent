"""
ESG Report Service
------------------
Backend Service for generating reports.
"""

from typing import Dict, Any, Optional
import os
from .generator import generate_esg_report

class ReportService:
    """
    Service class to handle report generation logic.
    Can be used by FastAPI endpoints or standard scripts.
    """
    
    def generate_html_report(self, data: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Generates HTML report from data.
        
        Args:
            data (Dict[str, Any]): The complete data dictionary (including esg_data.json content + AI generated content)
            output_path (Optional[str]): If provided, saves the HTML to this path.
            
        Returns:
            str: The raw HTML content.
        """
        # Validate critical fields? (Optional, logic is in generator)
        
        # Generate
        report_html = generate_esg_report(data)
        
        # Save if requested
        if output_path:
             # Ensure directory exists
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_html)
            print(f"[ReportService] Report saved to: {output_path}")
            
        return report_html

# Singleton instance if needed
report_service = ReportService()
