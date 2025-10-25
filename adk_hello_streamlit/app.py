import streamlit as st
import asyncio
from multi_tool_agent.agent import root_agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
import os
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# 🔑 Load environment variables
# ---------------------------------------------------------------------
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")
print("✅ GOOGLE_API_KEY loaded successfully:", os.getenv("GOOGLE_API_KEY")[:6] + "****")

# ---------------------------------------------------------------------
# 🧠 Setup ADK session and runner
# ---------------------------------------------------------------------
session_service = InMemorySessionService()
APP_NAME = "weather_app"
USER_ID = "streamlit_user"
SESSION_ID = "session_streamlit"

session = asyncio.run(session_service.create_session(
    app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
))
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

# ---------------------------------------------------------------------
# ⚙️ Helper: call agent asynchronously
# ---------------------------------------------------------------------
def call_agent(query: str) -> str:
    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response = "No response."

    async def _run():
        nonlocal final_response
        async for event in runner.run_async(
            user_id=USER_ID, session_id=SESSION_ID, new_message=content
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response = event.content.parts[0].text
                break

    asyncio.run(_run())
    return final_response

# ---------------------------------------------------------------------
# 🎨 Streamlit Dark Themed UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="ADK Weather Agent", page_icon="🌤️", layout="centered")

# Inject dark theme styling
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #0e1117;
        color: #f0f0f0;
    }
    .stTextInput > div > div > input {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .stButton button {
        background: linear-gradient(90deg, #0072ff, #00c6ff);
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #0059b3, #00a3cc);
    }
    .chat-bubble {
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        width: fit-content;
        max-width: 80%;
    }
    .user-bubble {
        background-color: #004d99;
        color: #ffffff;
        margin-left: auto;
        text-align: right;
    }
    .agent-bubble {
        background-color: #1c1f26;
        border: 1px solid #2c2f36;
        color: #f2f2f2;
        margin-right: auto;
    }
    .header {
        text-align: center;
        background: linear-gradient(90deg, #004e92, #000428);
        padding: 1.5rem;
        border-radius: 12px;
        color: #e3f2fd;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# 🧭 Header
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="header">
        <h2>🌤️ ADK Weather Assistant</h2>
        <p style="color:#b0bec5;">Ask me about the weather in any city!</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# 💬 Chat container
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages
for msg in st.session_state.messages:
    role = msg["role"]
    bubble_class = "user-bubble" if role == "user" else "agent-bubble"
    icon = "🧑‍💻" if role == "user" else "🤖"
    st.markdown(
        f"""
        <div class="chat-bubble {bubble_class}">
            <b>{icon} {'You' if role=='user' else 'Agent'}:</b><br>{msg['content']}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
# ✏️ Input and Response
# ---------------------------------------------------------------------
prompt = st.chat_input("Ask about weather...")

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"<div class='chat-bubble user-bubble'>🧑‍💻 You:<br>{prompt}</div>", unsafe_allow_html=True)

    # Get agent response
    with st.spinner("🤖 Thinking..."):
        try:
            answer = call_agent(prompt)
        except Exception as e:
            answer = f"⚠️ Error: {e}"

    # Determine weather emoji
    icon = "☀️" if "sunny" in answer.lower() else "🌧️" if "rain" in answer.lower() else "🌤️"
    agent_reply = f"{icon} {answer}"

    # Show and store
    st.markdown(
        f"<div class='chat-bubble agent-bubble'>🤖 Agent:<br>{agent_reply}</div>",
        unsafe_allow_html=True,
    )
    st.session_state.messages.append({"role": "assistant", "content": agent_reply})

# ---------------------------------------------------------------------
# 🧩 Footer
# ---------------------------------------------------------------------
st.markdown(
    """
    <hr style="margin-top:1.5rem;">
    <p style='text-align:center; color:#607d8b;'>Built with ❤️ using Streamlit + Google ADK</p>
    """,
    unsafe_allow_html=True,
)
