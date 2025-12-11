import React, { useState, useCallback, useEffect, useMemo } from "react"
import Sidebar from "./components/Sidebar"
import MainContent from "./components/MainContent"
import ChatBotPanel from "./components/ChatBotPanel"
import "./App.css"
import { GUIDE_CONVERSATION, GUIDE_CONVERSATION_ID } from "./constants/conversations"

const API_BASE = "http://localhost:8000/api"

function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [conversations, setConversations] = useState([])
  const [activeConversationId, setActiveConversationId] = useState(null)
  const [filesByConversation, setFilesByConversation] = useState({})

  const fetchConversations = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/conversations`)
      if (!response.ok) {
        throw new Error("failed to load conversations")
      }
      const data = await response.json()
      setConversations(data)
      return data
    } catch (err) {
      console.error("대화방 목록을 불러오지 못했습니다.", err)
      setConversations([])
      return []
    }
  }, [])

  const handleCreateConversation = useCallback(async () => {
    // "새 채팅" 클릭 시 새 대화방을 만들고 즉시 선택
    try {
      const response = await fetch(`${API_BASE}/conversations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      })
      if (!response.ok) {
        throw new Error("failed to create conversation")
      }
      const data = await response.json()
      await fetchConversations()
      setActiveConversationId(data.id)
      return data
    } catch (err) {
      console.error("새 대화를 생성하지 못했습니다.", err)
      return null
    }
  }, [fetchConversations])

  useEffect(() => {
    // 최초 진입 시 대화방 목록을 불러오고, 없다면 가이드만 표시
    const bootstrap = async () => {
      const list = await fetchConversations()
      if (list.length === 0) {
        const created = await handleCreateConversation()
        if (created) {
          setActiveConversationId(created.id)
        }
      } else if (!activeConversationId) {
        setActiveConversationId(list[0].id)
      }
    }
    bootstrap()
  }, [activeConversationId, fetchConversations, handleCreateConversation])

  useEffect(() => {
    // 대화방 목록이 갱신됐는데 선택된 ID가 없으면 최신 것을 기본 선택
    if (!activeConversationId && conversations.length > 0) {
      setActiveConversationId(conversations[0].id)
    }
  }, [activeConversationId, conversations])

  const handleConversationUpdated = useCallback(() => {
    fetchConversations()
  }, [fetchConversations])

  const fetchConversationFiles = useCallback(async (conversationId) => {
    if (!conversationId || conversationId === GUIDE_CONVERSATION_ID) {
      setFilesByConversation((prev) => ({ ...prev, [conversationId || "guide"]: [] }))
      return []
    }
    try {
      const response = await fetch(`${API_BASE}/conversations/${conversationId}/files`)
      if (!response.ok) {
        throw new Error("failed to load files")
      }
      const data = await response.json()
      setFilesByConversation((prev) => ({ ...prev, [conversationId]: data }))
      return data
    } catch (err) {
      console.error("파일 목록을 불러오지 못했습니다.", err)
      setFilesByConversation((prev) => ({ ...prev, [conversationId]: [] }))
      return []
    }
  }, [])

  const handleDeleteConversation = useCallback(async (conversationId) => {
    if (!conversationId || conversationId === GUIDE_CONVERSATION_ID) {
      if (conversationId === GUIDE_CONVERSATION_ID) {
        alert("가이드는 삭제할 수 없습니다.")
      }
      return
    }
    const confirmDelete = window.confirm("선택한 대화를 삭제할까요?")
    if (!confirmDelete) return
    try {
      const response = await fetch(`${API_BASE}/conversations/${conversationId}`, {
        method: "DELETE",
      })
      if (!response.ok) {
        throw new Error("failed to delete conversation")
      }
      const updated = await fetchConversations()
      if (conversationId === activeConversationId) {
        if (updated.length > 0) {
          setActiveConversationId(updated[0].id)
        } else {
          const created = await handleCreateConversation()
          setActiveConversationId(created?.id || GUIDE_CONVERSATION_ID)
        }
      }
      setFilesByConversation((prev) => {
        const next = { ...prev }
        delete next[conversationId]
        return next
      })
    } catch (err) {
      console.error("대화를 삭제하지 못했습니다.", err)
    }
  }, [activeConversationId, fetchConversations, handleCreateConversation])

  const sidebarConversations = useMemo(() => {
    return [...conversations, GUIDE_CONVERSATION]
  }, [conversations])

  useEffect(() => {
    if (activeConversationId) {
      fetchConversationFiles(activeConversationId)
    }
  }, [activeConversationId, fetchConversationFiles])

  return (
    <div className="app-shell">
      <button
        onClick={() => setIsSidebarOpen((prev) => !prev)}
        className="sidebar-toggle"
      >
        {isSidebarOpen ? "◀" : "▶"}
      </button>
      <Sidebar
        isOpen={isSidebarOpen}
        conversations={sidebarConversations}
        activeConversationId={activeConversationId}
        onSelectConversation={setActiveConversationId}
        onCreateConversation={handleCreateConversation}
        onDeleteConversation={handleDeleteConversation}
        files={filesByConversation[activeConversationId] || []}
        conversationId={activeConversationId === GUIDE_CONVERSATION_ID ? null : activeConversationId}
        onFilesRefresh={() => fetchConversationFiles(activeConversationId)}
      />
      <div className={`central-panel ${isSidebarOpen ? "" : "expanded"}`}>
        <MainContent activeConversationId={activeConversationId} />
      </div>
      <ChatBotPanel
        conversationId={activeConversationId}
        onConversationUpdated={handleConversationUpdated}
        onConversationChange={setActiveConversationId}
      />
    </div>
  )
}

export default App
