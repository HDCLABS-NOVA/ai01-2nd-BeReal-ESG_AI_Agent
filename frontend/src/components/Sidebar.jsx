import React from "react"
import "./Sidebar.css"
import fileIcon from "../assets/file_icon.png"
import messageIcon from "../assets/message_icon.png"
import FileUploader from "./FileUploader"

function Sidebar({
  isOpen,
  conversations = [],
  activeConversationId,
  onSelectConversation,
  onCreateConversation,
  onDeleteConversation,
  conversationId,
  files = [],
  onFilesRefresh,
}) {

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
        <button className="new-chat-btn" onClick={onCreateConversation}>
          + 새 채팅
        </button>
        <div className="chat-history-list">
          {conversations.length === 0 ? (
            <p className="chat-history-empty">아직 대화가 없습니다.</p>
          ) : (
            conversations.map((conversation) => (
              <div
                key={conversation.id}
                className={`chat-history-item ${conversation.isGuide ? "guide" : ""} ${activeConversationId === conversation.id ? "active" : ""}`}
                onClick={() => onSelectConversation && onSelectConversation(conversation.id)}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault()
                    onSelectConversation && onSelectConversation(conversation.id)
                  }
                }}
              >
                <span className="chat-history-title">{conversation.title || "새 대화"}</span>
                {!conversation.isGuide && (
                  <button
                    type="button"
                    className="chat-history-delete"
                    onClick={(event) => {
                      event.stopPropagation()
                      onDeleteConversation && onDeleteConversation(conversation.id)
                    }}
                  >
                    ✕
                  </button>
                )}
              </div>
            ))
          )}
        </div>
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
          <FileUploader
            conversationId={conversationId}
            files={files}
            onUploadComplete={onFilesRefresh}
          />
        </div>
      </div>
    </div>
  )
}

export default Sidebar
