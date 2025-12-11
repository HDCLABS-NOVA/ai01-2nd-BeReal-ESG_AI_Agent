import logging
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
    conversation_id: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    try:
        if conversation_id:
            conversation = agent_manager.get_conversation(conversation_id)
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_text = _extract_text_from_file(file_path, file.content_type)
        size_bytes = os.path.getsize(file_path)
        if conversation_id:
            agent_manager.add_conversation_file(
                conversation_id,
                filename=file.filename,
                path=file_path,
                size_bytes=size_bytes,
                text=file_text,
            )
        else:
            # Legacy: 전역 uploaded_files 리스트만 갱신
            current_files = agent_manager.get_context().get("uploaded_files", [])
            filtered = [entry for entry in current_files if entry.get("filename") != file.filename]
            relative_path = f"/static/uploads/{file.filename}"
            filtered.append({"filename": file.filename, "path": relative_path})
            if len(filtered) > 50:
                filtered = filtered[-50:]
            agent_manager.update_context("uploaded_files", filtered)

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
        rag_snippets = agent_manager.retrieve_conversation_snippets(conversation_id, request.query)
        rag_text = "\n\n".join(rag_snippets) if rag_snippets else "None"
        rag_text = "\n\n".join(rag_snippets) if rag_snippets else "None"
        rag_text = "\n\n".join(rag_snippets) if rag_snippets else "None"
        rag_text = "\n\n".join(rag_snippets) if rag_snippets else "None"
        rag_text = "\n\n".join(rag_snippets) if rag_snippets else "None"
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

        [Retrieved Segments from Uploaded Files]
        {rag_text}
        
        [Guidelines]
        - 답변 형식은 유연하게 사용자가 이해하기 쉬운 Markdown으로 작성하되, 필요 시 요약/근거/권고 구조를 활용하라.
        - Regulation 관련 질문에는 최신 규제 업데이트를 우선 반영하라.
        - 업로드된 파일이나 검색된 세그먼트에서 중요 근거가 있으면 인용해 설명하라.
        - 중요한 숫자·지표·정책명은 굵게 표시해 주목성을 높여라.
        - 모르는 내용은 솔직하게 밝히고 어떤 에이전트를 실행해야 할지 제안해라.
        - 기본 언어는 한국어이지만, 사용자가 영어로 질문하면 동일 언어로 답하라.
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
        rag_snippets = agent_manager.retrieve_conversation_snippets(conversation_id, request.query)
        rag_text = "\n\n".join(rag_snippets) if rag_snippets else "None"

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

        [Retrieved Segments from Uploaded Files]
        {rag_text}

        [Guidelines]
        - 질문 의도에 맞춰 유연하게 Markdown을 사용하되, 필요하면 요약/근거/권고 등으로 자연스럽게 나눠라.
        - Regulation 관련 질문에는 최신 규제 업데이트를 우선적으로 언급하라.
        - 업로드 파일/검색된 세그먼트에서 나온 핵심 증거를 우선 인용하라.
        - 주요 수치나 정책명은 **굵게** 표시해 강조하고, 근거가 부족하면 솔직히 말하고 어떤 에이전트를 호출해야 할지 제안하라.
        - 기본 언어는 한국어이며, 사용자가 영어로 질문하면 영어로 답하라.
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
LOGGER = logging.getLogger(__name__)
