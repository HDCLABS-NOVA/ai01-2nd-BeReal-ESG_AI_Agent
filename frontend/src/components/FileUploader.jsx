import { useState, useRef } from "react"
import fileIcon from "../assets/file_icon.png"

export default function FileUploader({ conversationId, files = [], onUploadComplete }) {
    const [isDragging, setIsDragging] = useState(false)
    const fileInputRef = useRef(null)

    const handleDragOver = (e) => {
        e.preventDefault()
        setIsDragging(true)
    }

    const handleDragLeave = () => {
        setIsDragging(false)
    }

    const handleDrop = async (e) => {
        e.preventDefault()
        setIsDragging(false)
        const droppedFiles = Array.from(e.dataTransfer.files)
        await uploadFiles(droppedFiles)
    }

    const handleFileSelect = async (e) => {
        const selectedFiles = Array.from(e.target.files)
        e.target.value = ""
        await uploadFiles(selectedFiles)
    }

    const uploadFiles = async (fileList) => {
        if (!conversationId) {
            alert("먼저 대화를 선택하거나 생성하세요.")
            return
        }

        for (const file of fileList) {
            const formData = new FormData()
            formData.append("conversation_id", conversationId)
            formData.append("file", file)

            try {
                const response = await fetch("http://localhost:8000/api/upload", {
                    method: "POST",
                    body: formData,
                })
                if (!response.ok) throw new Error("Upload failed")
            } catch (error) {
                console.error("Error uploading file:", error)
            }
        }

        if (onUploadComplete) {
            onUploadComplete()
        }
    }

    return (
        <div className="flex flex-col h-full text-slate-900">
            <div
                className={`border-2 border-dashed rounded-xl p-6 text-center transition-all cursor-pointer ${isDragging
                    ? "bg-purple-200/30 border-purple-400 shadow-glow"
                    : "bg-white/70 border-white/80 hover:border-moonlightPurple/40"}
                    `}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
            >
                <input
                    type="file"
                    ref={fileInputRef}
                    className="hidden"
                    multiple
                    onChange={handleFileSelect}
                />
                <div className="flex items-center justify-center gap-2">
                    <img src={fileIcon} alt="Files" className="w-6 h-6 object-contain" />
                    <p className="text-slate-900 text-lg font-semibold">Drag & Drop your files here</p>
                </div>
                <p className="text-slate-600 text-sm mt-2">or click to upload</p>
            </div>

            <div className="mt-6 flex-1 overflow-y-auto">
                <h3 className="font-semibold text-slate-900 mb-2">Uploaded Files</h3>
                <ul className="space-y-2">
                    {files.map((file) => (
                        <li key={file.id || file.filename} className="bg-white/80 p-2 rounded text-sm flex justify-between items-center shadow-sm">
                            <span className="truncate">{file.filename || file.name}</span>
                            <span className="text-slate-500 text-xs">
                                {file.size_bytes ? (file.size_bytes / 1024).toFixed(1) : file.size ? (file.size / 1024).toFixed(1) : "0.0"} KB
                            </span>
                        </li>
                    ))}
                    {files.length === 0 && (
                        <li className="text-slate-500 text-sm">업로드된 파일이 없습니다.</li>
                    )}
                </ul>
            </div>
        </div>
    )
}
