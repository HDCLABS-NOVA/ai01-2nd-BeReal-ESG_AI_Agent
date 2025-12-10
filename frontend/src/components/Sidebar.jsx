import React, { useState } from "react"
import "./Sidebar.css"
import fileIcon from "../assets/file_icon.png"
import messageIcon from "../assets/message_icon.png"
import FileUploader from "./FileUploader"

function Sidebar({ isOpen }) {
  const [uploadedFiles, setUploadedFiles] = useState([])

  const handleUpload = (newFiles) => {
    setUploadedFiles((prev) => [...prev, ...newFiles])
  }

  return (
    <div className={`sidebar-wrapper ${isOpen ? "open" : "closed"}`}>
      <div className="sidebar-card">
        <div className="sidebar-heading">
          <img src={messageIcon} alt="guide" />
          <div>
            <p className="subtitle">Guide</p>
            <h3>ESG 챗봇 기록</h3>
          </div>
        </div>
        <button className="chat-guide" onClick={() => window.dispatchEvent(new CustomEvent("showSample"))}>
          ✅ ESG 웹 사용 가이드 (기본 대화)
        </button>
      </div>

      <div className="sidebar-card" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <div className="sidebar-heading">
          <img src={fileIcon} alt="upload" />
          <div>
            <p className="subtitle">Upload</p>
            <h3>파일 업로드</h3>
          </div>
        </div>
        <div className="upload-box" style={{ flex: 1, border: 'none', background: 'transparent' }}>
          <FileUploader onUpload={handleUpload} files={uploadedFiles} />
        </div>
      </div>
    </div>
  )
}

export default Sidebar
