from pathlib import Path
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.tools import load_mcp_tools
from agents import PipelineState, init_mcp_tools, build_graph
from PIL import Image
import sys, os, json, uuid, asyncio, time, base64
import streamlit as st


load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()

ICON_PATH = BASE_DIR / "assets" / "page.png"
if ICON_PATH.exists():
  ICON = Image.open(ICON_PATH)
else:
  ICON = "💡"
st.set_page_config(page_title="Noma", page_icon=ICON, layout="centered")

st.markdown(
    """
    <style>
    @font-face {
      font-family: 'YeogiOttaeJalnan';
      src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_four@1.2/JalnanOTF00.woff') format('woff');
      font-weight: normal; font-display: swap;
    }

    :root{
      --bg:#ffffff;
      --panel:#ffffff;
      --bubble-assist:#f6f7fb;
      --bubble-user:#efe9ff;
      --text:#0f1220;
      --muted:#616a7a;
      --accent:#A06BFF; 
      --accent-2:#E8A6FF; 
      --border:rgba(10,20,40,0.12);
      --shadow: 0 8px 24px rgba(0,0,0,0.08);
      --radius: 18px;
      --radius-sm: 12px;
      --maxw: 760px;
      --pagew: 860px;
    }

    @media (prefers-color-scheme: dark){
      :root{
        --bg:#0f1220;
        --panel:#14182b;
        --bubble-assist:#171c34;
        --bubble-user:#1d1537;
        --text:#E6E8EF;
        --muted:#9aa3b2;
        --border:rgba(255,255,255,0.08);
        --shadow: 0 8px 24px rgba(0,0,0,0.32);
      }
    }

    body, .stApp {
      font-family: 'Noto Sans KR', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .main .block-container{
      max-width: var(--pagew);
      padding-top: 1rem;
      padding-bottom: 2.5rem;
      margin-left: auto; margin-right: auto;
    }

    [data-testid="stSidebar"]{
      background: var(--panel) !important;
      border-right: 1px solid var(--border);
    }

    h1.app-title{
      font-family: 'YeogiOttaeJalnan', ui-sans-serif, system-ui;
      font-weight: 800; letter-spacing: -0.02em;
      margin: 0.2rem 0 0.35rem 0;
      font-size: clamp(1.8rem, 3vw, 2.4rem);
      background: linear-gradient(90deg,var(--accent),var(--accent-2));
      -webkit-background-clip: text; background-clip: text; color: transparent;
      text-align: center;
    }
    p.app-caption{
      margin: 0 0 0.9rem 0; color: var(--muted); font-size: 0.98rem; text-align: center;
    }

    [data-testid="stChatMessage"]{
      background: var(--bubble-assist);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 12px 14px;
      margin: 10px auto;
      box-shadow: var(--shadow);
      width: 100%;
      max-width: var(--maxw);
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] svg[data-testid="user-avatar"]),
    [data-testid="stChatMessage"][data-testid*="user"]{
      background: var(--bubble-user);
      border: 1px solid rgba(160,107,255,0.35);
    }

    [data-testid="stChatMessage"] pre, [data-testid="stChatMessage"] code{
      background: #0c1020 !important; color: #E6E8EF !important;
      border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important;
    }

    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='app-title'>Noma</h1>", unsafe_allow_html=True)
st.markdown("<p class='app-caption'>데이터로 완성하는 당신만의 AI 마케팅 비서</p>", unsafe_allow_html=True)
st.markdown("<p class='app-caption'>가맹점명과 요구사항을 입력하면 분석을 시작합니다!</p>", unsafe_allow_html=True)
st.markdown("<p class='app-caption'>(예: 육육**의 주요 방문고객을 기반으로 마케팅 채널을 추천하고 홍보안을 작성해줘)</p>", unsafe_allow_html=True)

def inject_background(image_path: str, mode: str = "light", opacity: float = 0.10):
    p = Path(image_path)
    if not p.exists():
        return
    data = base64.b64encode(p.read_bytes()).decode()
    prefix = f"@media (prefers-color-scheme: {mode}) {{"
    suffix = "}"
    css = f"""
    <style>
    {prefix}
    .stApp::before {{
        content:"";
        position: fixed; inset: 0;
        background-image: url("data:image/png;base64,{data}");
        background-size: cover; background-position: center; background-repeat: no-repeat;
        opacity: {opacity}; z-index: 0; pointer-events: none;
    }}
    .stApp > div:first-child {{ position: relative; z-index: 1; }}
    {suffix}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_background("assets/bg_d.png", mode="dark", opacity=0.10)
inject_background("assets/bg_w.jpg", mode="light", opacity=0.40)


API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY 가 설정되어 있지 않습니다.")
    st.stop()


with open(BASE_DIR / "settings" / "config.json", "r", encoding="utf-8") as fr:
    CONFIG = json.load(fr)

MODEL_ID = CONFIG.get("model", "gemini-2.5-flash")
server_params = StdioServerParameters(command=sys.executable, args=[str(BASE_DIR / "server.py")])

def get_llms():
    return {
        "data_analyzer": ChatGoogleGenerativeAI(
            model=MODEL_ID, google_api_key=API_KEY, **CONFIG["hyperparams"]["data_analyzer"]
        ),
        "goal_setter": ChatGoogleGenerativeAI(
            model=MODEL_ID, google_api_key=API_KEY, **CONFIG["hyperparams"]["goal_setter"]
        ),
        "marketer": ChatGoogleGenerativeAI(
            model=MODEL_ID, google_api_key=API_KEY, **CONFIG["hyperparams"]["marketer"]
        ),
        "summarizer": ChatGoogleGenerativeAI(
            model=MODEL_ID, google_api_key=API_KEY, **CONFIG["hyperparams"]["summarizer"]
        ),
    }

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
            text = final_state.get("final_output", "죄송합니다. 답변을 생성하지 못했습니다.")
            return {"text": text}


def load_avatar(default_emoji: str, asset_path: str | None):
    if asset_path:
        p = Path(asset_path)
        if p.exists():
            try:
                return p.read_bytes()
            except Exception:
                pass
    return default_emoji

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}]
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.bot_avatar = load_avatar("🤖", "assets/bot.png")
    st.session_state.user_avatar = load_avatar("🧑", "assets/user.jpg") 
    st.session_state.last_exec_time = None

with st.sidebar:
    st.subheader("대화 관리")
    if st.button("새 대화 시작", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "새로운 대화를 시작합니다."}]
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.last_exec_time = None
        st.rerun()

for m in st.session_state.messages:
    role = "user" if m["role"] == "user" else "assistant"
    avatar = st.session_state.get("user_avatar" if role == "user" else "bot_avatar", "🤖")
    with st.chat_message(role, avatar=avatar):
        st.markdown(m["content"])

if st.session_state.last_exec_time is not None:
    st.caption(f"⏱️ 추론 시간: {st.session_state.last_exec_time:.2f}초")

prompt = st.chat_input("입력을 기다리고 있어요...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=st.session_state.get("user_avatar", "🧑")):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=st.session_state.get("bot_avatar", "🤖")):
        output_placeholder = st.empty()
        with st.spinner("🤔 생각 중..."):
            t0 = time.monotonic()
            result = asyncio.run(
                call_with_mcp(
                    prompt_text=prompt,
                    thread_id=st.session_state.thread_id,
                )
            )
            t1 = time.monotonic()

        text = result.get("text", "오류가 발생했습니다.")
        st.session_state.last_exec_time = t1 - t0
        output_placeholder.write_stream(typewriter(text, chunk_size=2, delay_sec=0.03))

        st.session_state.messages.append({"role": "assistant", "content": text})
