import React, { useState } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import html2pdf from "html2pdf.js"
import logo from "/B_clean2.png"
import "./MainContent.css"
import { GUIDE_CONVERSATION_ID, GUIDE_REPORTS } from "../constants/conversations"

const API_BASE = "http://localhost:8000/api"

function MainContent({ activeConversationId }) {
  const [reports, setReports] = useState([])
  const [search, setSearch] = useState("")
  const isGuideMode = activeConversationId === GUIDE_CONVERSATION_ID

  React.useEffect(() => {
    const fetchReports = async () => {
      if (!activeConversationId || isGuideMode) return
      try {
        const res = await fetch(`${API_BASE}/conversations/${activeConversationId}/reports`)
        if (res.ok) {
          const data = await res.json()
          // Backend returns list of reports.
          // We format them if needed, or just use as is. 
          // The report structure in backend: { id, title, content, items, ... }
          // Frontend expects same.
          // Sort by created_at desc if needed, but let's assume backend/frontend consistent.
          // Actually backend list_conversation_reports returns list.
          // We should setReports(data). 
          // But we also have real-time updates.
          // The real-time updates append to the list. 
          // If we fetch here, we might overwrite real-time updates if they happen simultaneously?
          // Real-time adds to state. Fetch initializes state.
          // So fetch should run on activeConversationId change.
          setReports(data.reverse()) // Show newest first? state prepend logic suggests newest first.
        }
      } catch (e) {
        console.error("Failed to fetch reports:", e)
      }
    }

    if (isGuideMode) {
      setReports([]) // Guide mode handled by render logic, but clear state just in case
    } else {
      setReports([]) // Clear previous conversation reports
      fetchReports()
    }

    const reportHandler = (e) => {
      const newReport = e.detail
      setReports(prev => [newReport, ...prev])
    }

    window.addEventListener("newReport", reportHandler)
    return () => {
      window.removeEventListener("newReport", reportHandler)
    }
  }, [activeConversationId, isGuideMode])

  const handleDownloadPDF = (report) => {
    const element = document.getElementById(`report-content-${report.id}`)
    if (!element) {
      alert("다운로드할 내용을 찾을 수 없습니다.")
      return
    }

    const opt = {
      margin: 10,
      filename: `${report.title}.pdf`,
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: { scale: 2 },
      jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
    }

    html2pdf().set(opt).from(element).save()
      .then(() => alert(`📄 "${report.title}" PDF 다운로드가 시작되었습니다.`))
      .catch((err) => {
        console.error(err)
        alert("PDF 생성 중 오류가 발생했습니다.")
      })
  }

  return (
    <div className="main-content">
      <div className="main-header">
        <div className="header-title">
          <img
            src={logo}
            alt="logo"
            onClick={() => window.location.reload()}
          />
          <div>
            <p>LLM Output</p>
            <h2>생성된 보고서 / 체크리스트</h2>
          </div>
        </div>
        <div className="header-actions">
          <div className="search-box">
            <span>🔍</span>
            <input
              type="text"
              placeholder="보고서 검색"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
        </div>
      </div>

      <div className="report-list">
        {(isGuideMode ? GUIDE_REPORTS : reports)
          .filter((report) => report.title.includes(search))
          .map((report) => (
            <div className="report-box" key={report.id}>
              <div className="report-header">
                <h3>{report.title}</h3>
                <button className="save-btn" onClick={() => handleDownloadPDF(report)}>
                  PDF 다운로드
                </button>
              </div>
              <div id={`report-content-${report.id}`}>
                {report.content ? (
                  <div className="report-markdown">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {report.content}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <ul>
                    {report.items && report.items.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          ))}
      </div>
    </div>
  )
}

export default MainContent
