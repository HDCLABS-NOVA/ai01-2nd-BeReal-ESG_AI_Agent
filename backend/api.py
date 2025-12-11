from fastapi import APIRouter, UploadFile, File, HTTPException, Body, Form
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import shutil
import os
from pathlib import Path


from src.tools.report_tool.report_tool import generate_report_from_query
from src.tools.regulation_tool import _monitor_instance as regulation_monitor
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
        You are an expert ESG AI Assistant. Provide concise, tailored answers that reflect the user's goal and constraints.

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
        - Start by tagging the user's goal/constraints in one line; if unclear, ask ONE short clarifying question, then proceed.
        - Use evidence in this priority: Regulation Updates → Policy Analysis → Risk Assessment → Report Draft → Uploaded Files → Chat History; if absent, note '해당 근거 없음'.
        - Keep internal reasoning to 3 short lines before responding.
        - Do not invent numbers/dates absent from context; flag missing data explicitly. When giving numbers, cite the source inline. If regulation/policy is mentioned, add a one-line note that this is not legal advice.
        - Tone: professional and friendly; keep sections 2–4 bullets/lines; keep the whole response concise (~200 words).
        - Language follows the user (default Korean); avoid mixing languages. Use - or * for bullets, **bold** for emphasis, `code` for technical terms.
        - If confidence is low, mark it (신뢰도: 높음/중간/낮음) and suggest what to check next (file/regulation/data).
        - ALWAYS use MARKDOWN formatting.

        [Output Format - keep structured but flexible]
        ## 🎯 목표/제약
        - (1줄; 모르면 질문 1개)

        ## 📊 요약
        - 2~3문장 핵심

        ## 🔍 근거 (신뢰도 표기)
        - 근거 1 (신뢰도: …)
        - 근거 2
        - 근거 3 또는 '추가 데이터 필요: ...'

        ## 💡 권고사항
        - 권고 1 (사용자 목표/제약 반영)
        - 권고 2

        ## ▶️ 다음 행동
        - 실행 제안 1~2개 + 필요한 확인사항 1개

        If you don't know, say so and recommend running the appropriate agent (Regulation, Policy, Risk, Report).
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
        # 1. Conversation Setup (User's Logic)
        conversation_id = request.conversation_id
        if conversation_id:
            conversation = agent_manager.get_conversation(conversation_id)
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
        else:
            conversation = agent_manager.create_conversation()
            conversation_id = conversation["id"]

        history = agent_manager.get_conversation_history(conversation_id)
        # 2. Intent Detection (Report Logic)
        # Check if the user specifically wants to *generate* or *create* a report/checklist/document.
        class IntentAnalysis(BaseModel):
            is_generation_request: bool = Field(description="True if user wants to CREATE/WRITE a report/checklist, False otherwise.")
            
        intent_system_prompt = """
        Analyze the user's latest query to determine if they want to GENERATE a new report, checklist, or document.
        
        True:
        - "Make a safety report"
        - "Generate a checklist for ESG"
        - "Write a draft"
        
        False:
        - "What is K-ESG?"
        - "Summarize this file"
        - "Explain the safety policy"
        
        Return JSON: {"is_generation_request": boolean}
        """
        try:
            intent_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            structured_intent = intent_llm.with_structured_output(IntentAnalysis)
            intent = structured_intent.invoke([
                SystemMessage(content=intent_system_prompt),
                HumanMessage(content=request.query)
            ])
            is_report_request = intent.is_generation_request
        except Exception as e:
            print(f"⚠️ Intent detection failed: {e}")
            is_report_request = False

        report_content = None
        report_error = None
        context = agent_manager.get_context()

        # 3. Report Generation (If requested)
        if is_report_request:
            print(f"📄 Report generation intent detected for: {request.query}")
            try:
                # Content Schema Definition
                class MaterialIssue(BaseModel):
                    name: str = Field(description="Name of the material issue")
                    impact: int = Field(description="Importance (0-100)")
                    financial: int = Field(description="Financial impact (0-100)")
                    isMaterial: bool = Field(description="Always True")

                class ReportSection(BaseModel):
                    title: str = Field(description="Section heading")
                    content: str = Field(description="Section content in markdown")

                class ReportContentGen(BaseModel):
                    company_name: str = Field(description="Exact Name of the company found in the [Uploaded File Content].")
                    esg_strategy: str = Field(description="Main ESG strategy sentence from the file.")
                    env_policy: Optional[str] = Field(description="Environmental policy summary (Standard K-ESG).")
                    social_policy: Optional[str] = Field(description="Social policy summary (Standard K-ESG).")
                    gov_structure: Optional[str] = Field(description="Governance structure summary (Standard K-ESG).")
                    material_issues: Optional[List[MaterialIssue]] = Field(default=None, description="List of material issues. Empty if custom format needed.")
                    custom_sections: List[ReportSection] = Field(description="Dynamic sections for specific topics.")

                # Extract context from files (Using NEW Conversation File Logic ideally, but falling back to global for safety/compatibility)
                # Ideally: agent_manager.get_conversation_files_with_text(conversation_id)
                # But kept simpler for now to match previous logic structure
                uploaded_files = context.get("uploaded_files", [])
                file_context_str = ""
                
                if uploaded_files:
                    print(f"📂 Processing {len(uploaded_files)} files for report context...")
                    import pypdf
                    for text_file in uploaded_files: 
                        try:
                            fname = text_file.get("filename")
                            fpath = os.path.join(UPLOAD_DIR, fname)
                            content = ""
                            if fname.lower().endswith(".pdf"):
                                reader = pypdf.PdfReader(fpath)
                                content = "\n".join([utils.extract_text() for utils in reader.pages])
                            else:
                                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                                    content = f.read()
                            file_context_str += f"\n=== File: {fname} ===\n{content[:100000]}\n" 
                        except Exception as e:
                            print(f"⚠️ Failed to read file {fname}: {e}")

                content_system_prompt = f"""
                You are an expert ESG consultant (K-ESG).
                User Query: "{request.query}"
                
                [Context]
                - Industry: Construction (Default)
                - K-ESG Guideline v2.0
                
                [Uploaded File Content]
                {file_context_str if file_context_str else "No uploaded files found."}
                
                [Instructions]
                Generate content based ONLY on the file content.
                If 'Specific Topic' (e.g. Safety), ignore standard policies and create 'custom_sections'.
                If 'General', fill standard fields.
                If User mentions Company Name, use it. Else extract from file.
                """
                
                llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
                structured_llm = llm.with_structured_output(ReportContentGen)
                report_data_obj = structured_llm.invoke([
                    SystemMessage(content=content_system_prompt),
                    HumanMessage(content="Generate the report content.")
                ])
                report_data = report_data_obj.model_dump()

                # Generate Markdown (using esg_report_generator)
                # We need to ensure 'uploaded_files' metadata is passed if needed, or just generate from data
                # Using the existing generate_report_from_query wrapper might be easier if it accepts data, 
                # but here we did it manually. Let's call the generator logic directly?
                # Actually, standard 'generate_report_from_query' does the whole thing.
                # To avoid duplicating logic, I am effectively re-implementing 'generate_report_from_query' logic inline here 
                # as per previous 'api.py' state.
                
                from src.tools.report_tool.esg_report_generator import generate_esg_report
                report_content = generate_esg_report(report_data, standard="K-ESG")
                report_error = None
                
            except Exception as e:
                print(f"❌ Report generation failed: {e}")
                import traceback
                traceback.print_exc()
                report_content = None
                report_error = f"Report Generation Error: {str(e)}"
        
        # 4. Standard Chat Context & Response
        custom_result = await agent_manager.run_custom_agent(request.query)
        risk_assessment = context.get('risk_assessment')
        risk_summary = str(risk_assessment)[:500] + "..." if risk_assessment else "None"
        
        file_summaries = agent_manager.list_conversation_files(conversation_id)
        file_context = agent_manager.build_file_context(conversation_id)
        file_names = [entry["filename"] for entry in file_summaries]

        system_prompt = f"""
        You are an expert ESG AI Assistant. Provide concise, tailored answers.

        [Current Context]
        - Uploaded Files: {file_names if file_names else 'None'}
        - Latest Regulation Updates: {str(context.get('regulation_updates'))[:500] + "..." if context.get('regulation_updates') else "None"}
        - Policy Analysis: {context.get('policy_analysis', 'None')}
        - Risk Assessment: {risk_summary}
        - Report Draft: {context.get('report_draft', 'None')}
        
        [Uploaded File Excerpts]
        {file_context if file_context else 'None'}

        [Instructions]
        - Tag user goal.
        - Use evidence priority: Regulation -> Policy -> Risk -> Report -> Files.
        - Tone: Professional, friendly, concise.
        - Language: Korean default.
        - ALWAYS use MARKDOWN.
        
        [Output Format]
        ## 🎯 목표/제약
        ## 📊 요약
        ## 🔍 근거
        ## 💡 권고사항
        ## ▶️ 다음 행동
        """
        
        if report_content:
             system_prompt += "\n[System Note]\nA report has just been generated and displayed to the user. Briefly mention this in your response."

        llm = ChatOpenAI(model="gpt-4o", temperature=0.5, streaming=True)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.query)
        ]

        agent_manager.append_conversation_message(conversation_id, "user", request.query)
        assistant_buffer = {"text": ""}

        async def event_generator():
            try:
                if report_content:
                    yield f"data: {json.dumps({'report': report_content})}\n\n"
                
                if report_error:
                    yield f"data: {json.dumps({'error': report_error})}\n\n"
                
                async for chunk in llm.astream(messages):
                    token = chunk.content or ""
                    if token:
                        assistant_buffer["text"] += token
                        yield f"data: {json.dumps({'token': token})}\n\n"
                
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
        import traceback
        traceback.print_exc()
        print(f"❌ [API Error] {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
