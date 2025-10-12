from pathlib import Path
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.tools import load_mcp_tools
from agents import PipelineState, init_mcp_tools, build_graph
import sys
import os
import json
import uuid
import asyncio
import time
import streamlit as st


load_dotenv()


st.set_page_config(page_title="Gemini Chat", page_icon="🎨", layout="centered")

st.markdown(
    """
    <style>
    @font-face {
    font-family: 'YeogiOttaeJalnan';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_four@1.2/JalnanOTF00.woff') format('woff');
    font-weight: normal;
    font-display: swap;
    }
    body, .stApp {
        font-family: 'Noto Sans KR', sans-serif;
        background-color: #1A1A2E;
        color: #E0E0E0;
    }
    h1.app-title{
        font-weight: 800; letter-spacing: -0.02em; margin: 0.2rem 0 0.4rem 0;
        font-size: clamp(1.8rem, 3vw, 2.4rem);
    }
    p.app-caption{ margin: 0 0 1.2rem 0; color: var(--muted); font-size: 0.98rem; }
    [data-testid="stSidebar"] { background: #161b22 !important; color: var(--text) !important; }    
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='app-title'>🎨 Gemini Custom Chat</h1>", unsafe_allow_html=True)
st.markdown("<p class='app-caption'>AI 에이전트와 대화하는 미니멀 채팅 앱</p>", unsafe_allow_html=True)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY 가 설정되어 있지 않습니다.")
    st.stop()

BASE_DIR = Path(__file__).parent.resolve()
with open(f"{BASE_DIR}/settings/config.json", "r", encoding="utf-8") as fr:
    CONFIG = json.load(fr)
MODEL_ID = CONFIG.get("model", "gemini-2.5-flash")
server_params = StdioServerParameters(command=sys.executable, args=[f"{BASE_DIR}/server.py"])


def get_llms():
    llms = {
        "data_analyzer": ChatGoogleGenerativeAI(
            model=MODEL_ID,
            google_api_key=API_KEY,
            **CONFIG["hyperparams"]["data_analyzer"],
        ),
        "goal_setter": ChatGoogleGenerativeAI(
            model=MODEL_ID,
            google_api_key=API_KEY,
            **CONFIG["hyperparams"]["goal_setter"],
        ),
        "marketer": ChatGoogleGenerativeAI(
            model=MODEL_ID, 
            google_api_key=API_KEY,
            **CONFIG["hyperparams"]["marketer"]
        ),
        "summarizer": ChatGoogleGenerativeAI(
            model=MODEL_ID,
            google_api_key=API_KEY,
            **CONFIG["hyperparams"]["summarizer"],
        ),
    }
    return llms


llms = get_llms()

def typewriter(text: str, chunk_size: int = 3, delay_sec: float = 0.03):
    buf = []
    for i, ch in enumerate(text):
        buf.append(ch)
        if (i + 1) % chunk_size == 0:
            yield "".join(buf)
            buf = []
            time.sleep(delay_sec)
    if buf:
        yield "".join(buf)
            
async def call_with_mcp(prompt_text: str, thread_id: str):
    init: PipelineState = {
        "user_query": prompt_text,
        "store_info": [],
        "request": [],
        "analysis": "",
        "marketing_problem": "",
        "reason_data": [],
        "marketing_goal": "",
        "goal_reason": "",
        "strategies": {},
        "reason": "",
        "final_output": "",
    }

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = init_mcp_tools(await load_mcp_tools(session), CONFIG["tools"])
            mcp_graph = build_graph(llms, tools)
            config = {"configurable": {"thread_id": thread_id}}
            final_state = await mcp_graph.ainvoke(init, config=config)
            text = final_state.get(
                "final_output", "죄송합니다. 답변을 생성하지 못했습니다."
            )
            return {"text": text}
        
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}
    ]
    st.session_state.thread_id = str(uuid.uuid4())

with st.sidebar:
    st.subheader("대화 관리")
    if st.button("새 대화 시작", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "새로운 대화를 시작합니다."}
        ]
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
        
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("메시지를 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        output_placeholder = st.empty()
        with st.spinner("🤔 생각 중..."):
            st_time = time.monotonic()
            result = asyncio.run(call_with_mcp(
                    prompt_text=prompt,
                    thread_id=st.session_state.thread_id,
                ))
            ed_time = time.monotonic()
        execution_time = ed_time - st_time
        text = result.get("text", "오류가 발생했습니다.")
        output_placeholder.write_stream(typewriter(text, chunk_size=2, delay_sec=0.03))
        
        st.caption(f"⏱️ 추론 시간: {execution_time:.2f}초")
        st.session_state.messages.append({"role": "assistant", "content": text})
