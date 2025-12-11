import React, { useState } from "react"
import logo from "/B_clean2.png"
import "./MainContent.css"
import { GUIDE_CONVERSATION_ID, GUIDE_REPORTS } from "../constants/conversations"

function MainContent({ activeConversationId }) {
  const [reports, setReports] = useState([])
  const [search, setSearch] = useState("")
  const isGuideMode = activeConversationId === GUIDE_CONVERSATION_ID

  React.useEffect(() => {
    const reportHandler = (e) => {
      const newReport = e.detail
      setReports(prev => [newReport, ...prev])
    }

    window.addEventListener("newReport", reportHandler)
    return () => {
      window.removeEventListener("newReport", reportHandler)
    }
  }, [])

  const handleSave = (reportTitle) => {
    alert(`📄 "${reportTitle}" 보고서를 저장했습니다.`)
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
                <button className="save-btn" onClick={() => handleSave(report.title)}>
                  저장
                </button>
              </div>
              <ul>
                {report.items.map((item, index) => (
                  <li key={index}>{item}</li>
                ))}
              </ul>
            </div>
          ))}
      </div>
    </div>
  )
}

export default MainContent
