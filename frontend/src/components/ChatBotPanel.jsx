import React, { useState, useRef } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import "./ChatBotPanel.css"
import messageIcon from "../assets/message_icon.png"
import uploadButton from "../assets/upload_button.png"
import { GUIDE_CONVERSATION_ID, GUIDE_MESSAGES } from "../constants/conversations"

const API_BASE = "http://localhost:8000/api"

function ChatBotPanel({ conversationId, onConversationUpdated, onConversationChange }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [historyError, setHistoryError] = useState(null)
  const textareaRef = useRef(null)
  const botIndexRef = useRef(null)
  const isGuideMode = conversationId === GUIDE_CONVERSATION_ID

  const fetchConversationHistory = React.useCallback(async (targetId) => {
    // 선택된 대화방의 전체 메시지를 백엔드에서 다시 불러와 동기화
    if (targetId === GUIDE_CONVERSATION_ID) {
      setMessages(GUIDE_MESSAGES)
      setHistoryError(null)
      return
    }
    if (!targetId) {
      setMessages([])
      setHistoryError(null)
      return
    }
    try {
      const response = await fetch(`${API_BASE}/conversations/${targetId}`)
      if (!response.ok) {
        throw new Error("failed to load history")
      }
      const data = await response.json()
      const formatted = (data.messages || []).map((entry) => ({
        sender: entry.role === "assistant" ? "bot" : "user",
        text: entry.content,
      }))
      setMessages(formatted)
      setHistoryError(null)
    } catch (err) {
      console.error("Failed to load conversation history", err)
      setMessages([])
      setHistoryError("대화 기록을 불러오지 못했습니다.")
    }
  }, [])

  React.useEffect(() => {
    fetchConversationHistory(conversationId)
  }, [conversationId, fetchConversationHistory])

  const handleSend = async () => {
    if (!input.trim()) return
    if (!conversationId) {
      // 대화방이 없는 상태에서 입력할 경우 UX 안내
      alert("새 채팅을 먼저 생성한 뒤 메시지를 입력하세요.")
      return
    }
    if (isGuideMode) {
      alert("샘플 가이드에서는 직접 메시지를 보낼 수 없습니다.")
      return
    }
    const userMessage = { sender: "user", text: input }
    const updatedMessages = [...messages, userMessage, { sender: "bot", text: "" }]
    const botIndex = updatedMessages.length - 1
    botIndexRef.current = botIndex
    setMessages(updatedMessages)
    const currentQuery = input
    setInput("")
    setIsLoading(true)
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
    }

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: currentQuery, conversation_id: conversationId }),
      })
      if (!response.ok || !response.body) {
        throw new Error("stream error")
      }
      const reader = response.body.getReader()
      const decoder = new TextDecoder("utf-8")
      let botText = ""
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        const events = chunk.split("\n\n")
        for (const event of events) {
          if (!event.trim()) continue
          const dataLine = event.split("data: ").pop()
          if (!dataLine) continue
          const payload = JSON.parse(dataLine)
          if (payload.token) {
            botText += payload.token
            setMessages((prev) => {
              const copy = [...prev]
              if (botIndexRef.current !== null && copy[botIndexRef.current]) {
                copy[botIndexRef.current] = { sender: "bot", text: botText }
              }
              return copy
            })
          }
          if (payload.done) {
            // 스트림 완료 시 최신 히스토리를 다시 불러와 프론트 상태를 서버와 일치시킴
            const latestConversationId = payload.conversation_id || conversationId
            if (payload.conversation_id && payload.conversation_id !== conversationId && onConversationChange) {
              onConversationChange(payload.conversation_id)
            }
            await fetchConversationHistory(latestConversationId)
            if (onConversationUpdated) {
              onConversationUpdated()
            }
          }
          if (payload.report) {
            // Signal MainContent to add this report
            // Simple logic: Use current query as title (or generic) and split items by newlines for now (since report is markdown)
            // Ideally, MainContent would parse the markdown. For now, let's pass the raw markdown.
            const reportData = {
              id: Date.now(),
              title: "새로 생성된 보고서", // Or derive from query if accessible
              content: payload.report, // Pass full content
              items: payload.report.split('\n').filter(line => line.trim().startsWith('-') || line.trim().startsWith('*')).map(l => l.replace(/^[-*]\s/, ''))
            }
            window.dispatchEvent(new CustomEvent("newReport", { detail: reportData }))
          }
          if (payload.error) {
            throw new Error(payload.error)
          }
        }
      }
    } catch (err) {
      console.error(err)
      setMessages((prev) => {
        const copy = [...prev]
        if (botIndexRef.current !== null && copy[botIndexRef.current]) {
          copy[botIndexRef.current] = { sender: "bot", text: "Error: 응답을 가져오지 못했습니다." }
          return copy
        }
        return [...prev, { sender: "bot", text: "Error: 응답을 가져오지 못했습니다." }]
      })
    } finally {
      setIsLoading(false)
      botIndexRef.current = null
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <div className="chat-title">
          <img src={messageIcon} alt="chat" />
          <div>
            <p>Assistant</p>
            <h3>ESG AI 챗봇</h3>
          </div>
        </div>
      </div>
      <div className="chat-window">
        <div className="chat-messages">
          {isGuideMode ? (
            <div className="chat-guide-mode">
              <div className="chat-guide-banner">
                <p>샘플 대화입니다. 왼쪽에서 "새 채팅"을 선택하면 실제 챗봇과 대화할 수 있습니다.</p>
              </div>
              {messages.map((msg, index) => (
                <div key={`guide-${index}`} className={`chat-message ${msg.sender}`}>
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={markdownComponents}
                  >
                    {msg.text}
                  </ReactMarkdown>
                </div>
              ))}
            </div>
          ) : (!conversationId || messages.length === 0) ? (
            <div className="chat-empty-state">
              {historyError || (conversationId ? "아직 메시지가 없습니다." : "좌측에서 새 채팅을 생성하세요.")}
            </div>
          ) : (
            messages.map((msg, index) => (
              <div key={index} className={`chat-message ${msg.sender}`}>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={markdownComponents}
                >
                  {msg.text || (msg.sender === "bot" && isLoading ? "" : msg.text)}
                </ReactMarkdown>
              </div>
            ))
          )}
        </div>
        <div className="chat-input">
          <textarea
            ref={textareaRef}
            placeholder="질문을 입력하세요..."
            rows={1}
            value={input}
            onChange={(e) => {
              setInput(e.target.value)
              if (textareaRef.current) {
                textareaRef.current.style.height = "auto"
                textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
              }
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                handleSend()
              }
            }}
          />
          <button className="send-btn" onClick={handleSend} disabled={!input.trim() || isLoading || !conversationId || isGuideMode}>
            <img src={uploadButton} alt="send" />
          </button>
        </div>
      </div>
    </div>
  )
}

export default ChatBotPanel
const markdownComponents = {
  h1: ({ node, ...props }) => (
    <h2 className="markdown-h1" {...props} />
  ),
  h2: ({ node, ...props }) => (
    <h3 className="markdown-h2" {...props} />
  ),
  h3: ({ node, ...props }) => (
    <h4 className="markdown-h3" {...props} />
  ),
  strong: ({ node, ...props }) => (
    <strong className="markdown-strong" {...props} />
  ),
  ul: ({ node, ordered, ...props }) => (
    <ul className="markdown-list" {...props} />
  ),
  ol: ({ node, ordered, ...props }) => (
    <ol className="markdown-list" {...props} />
  ),
  code: ({ node, inline, ...props }) => (
    inline ? <code className="markdown-code" {...props} /> : <code className="markdown-code-block" {...props} />
  ),
  pre: ({ node, ...props }) => (
    <pre className="markdown-pre" {...props} />
  ),
}
