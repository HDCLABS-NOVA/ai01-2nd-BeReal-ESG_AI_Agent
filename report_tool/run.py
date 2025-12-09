"""
ESG Report Tool - Runner
------------------------
Entry point to generate a report using the new backend service structure.
"""

import sys
import os
import json

# Add current directory to sys.path so we can import backend
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.reporting.service import report_service

def load_data(filename="esg_data.json"):
    """Load data from JSON file."""
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"[Warning] {filename} not found. Using empty data.")
    return {}

def main():
    print("[Report Tool] Starting Report Generation...")
    
    # 1. Load Data (Simulating Database or AI Output)
    data = load_data()
    
    # 2. Check for minimal data (optional, for demo)
    if not data:
        print(" -> No data found. Please ensure 'esg_data.json' exists.")
        # Minimal dummy data to prevent crash
        data = {
            "company_name": "Demo Company",
            "report_year": "2025"
        }
    
    # 3. Generate Report
    output_file = "esg_report.html"
    try:
        html = report_service.generate_html_report(data, output_path=output_file)
        print(f"[Success] Report generated at: {os.path.abspath(output_file)}")
        print(f" -> Size: {len(html)} bytes")
    except Exception as e:
        print(f"[Error] Failed to generate report: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
