from fastapi import APIRouter, UploadFile, File, HTTPException, Body, Form
from fastapi.responses import StreamingResponse
from typing import List, Optional
from pydantic import BaseModel
import shutil
import os
from pathlib import Path

from backend.manager import agent_manager

try:
    from PyPDF2 import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ChatRequest(BaseModel):
    query: str
    agent_type: Optional[str] = "general"
    conversation_id: Optional[str] = None

class AgentRequest(BaseModel):
    query: str
    focus_area: Optional[str] = None  # 리스크 도구 등에서 안전/환경 등 영역을 지정할 때 사용
    audience: Optional[str] = None  # 보고서 초안 대상 (경영진, 이사회 등)

class ConversationCreateRequest(BaseModel):
    title: Optional[str] = None

def _extract_text_from_file(file_path: str, content_type: Optional[str] = None) -> str:
    ext = Path(file_path).suffix.lower()
    try:
        if ext == ".pdf" and PdfReader is not None:
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                texts = []
                for page in reader.pages:
                    try:
                        texts.append(page.extract_text() or "")
                    except Exception:
                        continue
                return "\n".join(texts)
        if ext in {".txt", ".md", ".csv", ".json"}:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        # fallback binary decode
        with open(file_path, "rb") as f:
            data = f.read()
            return data.decode("utf-8", errors="ignore")
    except Exception as exc:
        print(f"[Upload] 텍스트 추출 실패 ({file_path}): {exc}")
        return ""


@router.post("/upload")
async def upload_file(
    conversation_id: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        conversation = agent_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_text = _extract_text_from_file(file_path, file.content_type)
        size_bytes = os.path.getsize(file_path)
        agent_manager.add_conversation_file(
            conversation_id,
            filename=file.filename,
            path=file_path,
            size_bytes=size_bytes,
            text=file_text,
        )

        return {
            "conversation_id": conversation_id,
            "filename": file.filename,
            "size_bytes": size_bytes,
            "status": "uploaded",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/context")
async def get_context():
    return agent_manager.get_context()

@router.get("/conversations")
async def list_conversations():
    # 대화방 목록(최근 업데이트 순)을 반환
    return agent_manager.list_conversations()

@router.post("/conversations")
async def create_conversation(request: ConversationCreateRequest):
    # 새 대화방을 만들고 UUID를 돌려줌
    return agent_manager.create_conversation(request.title)

@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    conversation = agent_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation

@router.get("/conversations/{conversation_id}/files")
async def list_conversation_files(conversation_id: str):
    conversation = agent_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return agent_manager.list_conversation_files(conversation_id)

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    if not agent_manager.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted", "conversation_id": conversation_id}

@router.post("/agent/{agent_type}")
async def run_agent(agent_type: str, request: AgentRequest):
    if agent_type == "policy":
        result = await agent_manager.run_policy_agent(request.query)
    elif agent_type == "regulation":
        result = await agent_manager.run_regulation_agent(request.query)
    elif agent_type == "risk":
        result = await agent_manager.run_risk_agent(request.query, request.focus_area)
    elif agent_type == "report":
        result = await agent_manager.run_report_agent(request.query, request.audience)
    elif agent_type == "custom":
        result = await agent_manager.run_custom_agent(
            request.query,
            focus_area=request.focus_area,
            audience=request.audience,
        )
    else:
        raise HTTPException(status_code=404, detail="Agent type not found")

    return {"result": result}

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json

@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        context = agent_manager.get_context()

        # 프론트에서 conversation_id를 보내면 해당 세션을 재사용
        conversation_id = request.conversation_id
        if conversation_id:
            conversation = agent_manager.get_conversation(conversation_id)
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
        else:
            # 없으면 새 대화를 만들어 ID를 발급
            conversation = agent_manager.create_conversation()
            conversation_id = conversation["id"]

        history = agent_manager.get_conversation_history(conversation_id)
        history_text = "\n".join(
            [
                f"User: {entry['content']}" if entry.get('role') == 'user' else f"Assistant: {entry['content']}"
                for entry in history
            ]
        )

        custom_result = await agent_manager.run_custom_agent(request.query)

        risk_assessment = context.get('risk_assessment')
        risk_summary = str(risk_assessment)[:500] + "..." if risk_assessment else "None"
        file_summaries = agent_manager.list_conversation_files(conversation_id)
        file_context = agent_manager.build_file_context(conversation_id)
        file_names = [entry["filename"] for entry in file_summaries]
        system_prompt = f"""
        You are an expert ESG AI Assistant. Your goal is to help the user with ESG (Environmental, Social, and Governance) related tasks.

        [Current Context]
        - Uploaded Files: {file_names if file_names else 'None'}
        - Latest Regulation Updates: {str(context.get('regulation_updates'))[:500] + "..." if context.get('regulation_updates') else "None"}
        - Policy Analysis: {context.get('policy_analysis', 'None')}
        - Risk Assessment: {risk_summary}
        - Report Draft: {context.get('report_draft', 'None')}
        
        [Conversation History]
        {history_text if history_text else 'None'}

        [Uploaded File Excerpts]
        {file_context if file_context else 'None'}
        
        [Instructions]
        - Answer the user's question based on the context provided above.
        - **IMPORTANT**: ALWAYS use MARKDOWN formatting for all responses
        - If the user asks about specific regulations or news, refer to the 'Latest Regulation Updates' section.
        - Be professional, concise, and helpful.

        [Output Format - MANDATORY]
        ## 📊 요약
        (2-3문장으로 핵심 내용을 명확하게 설명)

        ## 🔍 근거
        - 근거 항목 1
        - 근거 항목 2
        - 근거 항목 3

        ## 💡 권고사항
        - 권고 항목 1
        - 권고 항목 2

        [Formatting Rules]
        - Use ## for main section headings
        - Use - or * for bullet points (NOT •)
        - Use **bold** for emphasis on key terms
        - Use `code` for technical terms or file names
        - Use proper line breaks between sections
        - If you don't know the answer, admit it and suggest running a specific agent (Regulation, Policy, Risk, etc.).
        - Language: Korean (unless the user asks in English).
        """
        
        # 3. Call LLM (GPT-4o)
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.query)
        ]

        # user/assistant 모두 서버 측에 기록
        agent_manager.append_conversation_message(conversation_id, "user", request.query)

        response_msg = await llm.ainvoke(messages)
        response_text = response_msg.content

        agent_manager.append_conversation_message(conversation_id, "assistant", response_text)

        return {"conversation_id": conversation_id, "response": response_text}
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    try:
        context = agent_manager.get_context()
        # SSE 스트림도 동일하게 conversation_id를 요구
        conversation_id = request.conversation_id
        if conversation_id:
            conversation = agent_manager.get_conversation(conversation_id)
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
        else:
            conversation = agent_manager.create_conversation()
            conversation_id = conversation["id"]

        history = agent_manager.get_conversation_history(conversation_id)
        history_text = "\n".join(
            [
                f"User: {entry['content']}" if entry.get('role') == 'user' else f"Assistant: {entry['content']}"
                for entry in history
            ]
        )

        custom_result = await agent_manager.run_custom_agent(request.query)
        risk_assessment = context.get('risk_assessment')
        risk_summary = str(risk_assessment)[:500] + "..." if risk_assessment else "None"
        file_summaries = agent_manager.list_conversation_files(conversation_id)
        file_context = agent_manager.build_file_context(conversation_id)
        file_names = [entry["filename"] for entry in file_summaries]

        system_prompt = f"""
        You are an expert ESG AI Assistant. Your goal is to help the user with ESG (Environmental, Social, and Governance) related tasks.

        [Current Context]
        - Uploaded Files: {file_names if file_names else 'None'}
        - Latest Regulation Updates: {str(context.get('regulation_updates'))[:500] + "..." if context.get('regulation_updates') else "None"}
        - Policy Analysis: {context.get('policy_analysis', 'None')}
        - Risk Assessment: {risk_summary}
        - Report Draft: {context.get('report_draft', 'None')}

        [Conversation History]
        {history_text if history_text else 'None'}

        [Auto-Generated Insights]
        - Policy Summary: {custom_result.get('policy')}
        - Regulation Update: {custom_result.get('regulation')}
        - Risk Analysis: {custom_result.get('risk')}
        - Report Draft: {custom_result.get('report')}

        [Uploaded File Excerpts]
        {file_context if file_context else 'None'}

        [Instructions]
        - Answer using the template below to emulate an expert ESG consultant.
        - **IMPORTANT**: ALWAYS use MARKDOWN formatting for all responses

        [Output Format - MANDATORY]
        ## 📊 요약
        (2-3문장으로 핵심 내용을 명확하게 설명)

        ## 🔍 근거
        - 근거 항목 1
        - 근거 항목 2
        - 근거 항목 3

        ## 💡 권고사항
        - 권고 항목 1
        - 권고 항목 2

        [Formatting Rules]
        - Use ## for main section headings with emojis (📊 요약, 🔍 근거, 💡 권고사항)
        - Use - or * for bullet points (NOT •)
        - Use **bold** for emphasis on key terms
        - Use `code` for technical terms or file names
        - Use proper line breaks between sections
        - 답변에 최신 규제/정책/리스크 정보를 자연스럽게 녹여라.
        - Be professional, concise, and helpful.
        - If you don't know the answer, admit it and suggest running a specific agent (Regulation, Policy, Risk, etc.).
        - Language: Korean (unless the user asks in English).
        """

        llm = ChatOpenAI(model="gpt-4o", temperature=0.5, streaming=True)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.query)
        ]

        agent_manager.append_conversation_message(conversation_id, "user", request.query)

        assistant_buffer = {"text": ""}

        async def event_generator():
            try:
                async for chunk in llm.astream(messages):
                    token = chunk.content or ""
                    if token:
                        assistant_buffer["text"] += token
                        yield f"data: {json.dumps({'token': token})}\n\n"
                # 최종 응답을 한 번만 저장하기 위해 버퍼 사용
                agent_manager.append_conversation_message(
                    conversation_id,
                    "assistant",
                    assistant_buffer["text"],
                )
                yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
